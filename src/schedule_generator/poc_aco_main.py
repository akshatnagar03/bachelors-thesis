from matplotlib import pyplot as plt
import numpy as np
from src.schedule_generator.poc_aco_v2 import (
    Job,
    JobShopProblem,
    ObjectiveFunction,
)
from src.schedule_generator.poc_aco_machine_assignment import FullJobShopProblem
from src.production_orders import Workstation, parse_data


def from_assigned_machine_to_jssp(aco_machine: FullJobShopProblem) -> JobShopProblem:
    jobs: dict[int, Job] = dict()

    for sub_job in aco_machine.sub_jobs:
        if sub_job.machine == -1:
            raise ValueError("Sub-job has not been assigned a machine.")
        jobs[sub_job.task_id] = sub_job

    jssp = JobShopProblem(
        jobs,
        machines=set(aco_machine.machine_key.values()),
        number_of_jobs=len(jobs) // 2,
        number_of_tasks=len(jobs),
    )

    setup_times = np.zeros((len(jobs), len(jobs)))
    job_keys = list(jobs.keys())
    reverse_machine_key = {v: k for k, v in aco_machine.machine_key.items()}
    for j1 in job_keys:
        for j2 in job_keys:
            j1_job = jobs[j1]
            j2_job = jobs[j2]

            # If the jobs are from the same product, there are no setup times
            if j1_job.product_id == j2_job.product_id:
                continue

            machine: Workstation = [
                station
                for station in aco_machine.data.workstations
                if station.name == reverse_machine_key[j1_job.machine]
            ][0]
            # If the jobs have different tastes, there is a setup time for both bottling and mixing
            if j1_job.station_settings["taste"] != j2_job.station_settings["taste"]:
                setup_times[j1 - 1, j2 - 1] += machine.minutes_changeover_time_taste

            # If the jobs are on a bottling line and have different bottle sizes, there is a setup time
            if (
                j1_job.station_settings["bottle_size"]
                != j2_job.station_settings["bottle_size"]
                and "bottling" in machine.name.lower()
            ):
                setup_times[j1 - 1, j2 - 1] += (
                    machine.minutes_changeover_time_bottle_size
                )

    jssp.set_setup_times(setup_times)
    return jssp


def assign_machines(aco_machine: FullJobShopProblem) -> FullJobShopProblem:
    for sub_job in aco_machine.sub_jobs:
        to_assign = list(sub_job.available_machines.keys())[0]
        sub_job.assign_machine(to_assign)
    return aco_machine


class FullACO:
    def __init__(
        self,
        problem: FullJobShopProblem,
        objective_function: ObjectiveFunction = ObjectiveFunction.MAKESPAN,
        *,
        n_ants: int = 10,
        n_iter: int = 50,
        seed: int = 1234,
        rho: float = 0.1,
        alpha: float = 0.1,
        beta: float = 2.0,
        q_zero: float = 0.9,
        tau_zero: float = 1.0 / (50 * 750.0),
        verbose: bool = False,
        with_local_search: bool = True,
    ):
        """Initializes the two-stage ACO algorithm.

        First part implemented according to the paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=585892

        Args:
            problem (JobShopProblem): the job shop problem to solve.
            n_ants (int, optional): number of ants to use, a good indication is (# of jobs)/2 based on
                [this](https://arxiv.org/ftp/arxiv/papers/1309/1309.5110.pdf). Defaults to 10.
            n_iter (int, optional): number of iterations (i.e. how many times global update will happen).
                Defaults to 50.
            seed (int, optional): seed for numpy.random. Defaults to 1234.
            rho (float, optional): parameter for local update, and how much pheromones we evaoprate. Defaults to 0.1.
            alpha (float, optional): paramter for global update, and how much the pheromones evaporate. Defaults to 0.1.
            beta (float, optional): parameter for the weight we put on the heuristic. Defaults to 2.0.
            q_zero (float, optional): paramter for how often we just choose the highest probability (exploitation). Defaults to 0.9.
            tau_zero (float, optional): parameter for initial pheromones levels, a good estimate for this is 1.0/(n * Z_best)
                where n is the number of jobs/tasks and Z_best is a rough approximation of the optimal objective value. Defaults to 1.0/(50 * 750.0).
            verbose (bool, optional): defines how much should be printed to stdout. Defaults to False.
            with_local_search (bool, optional): defines if local search should be used. Defaults to True.
        """
        self.problem = problem
        self.n_ants = n_ants
        self.n_iter = n_iter
        np.random.seed(seed)
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.q_zero = q_zero
        self.tau_zero = tau_zero
        self.verbose = verbose
        self.with_local_search = with_local_search
        self.objective_function = objective_function

        # pheromones will be (tasks + 1) x (tasks + 1) matrix, since 0 is the
        # starting node and we start counting tasks from 1, so we need to add 1
        # to the number of tasks. However, (0,0) in pheromones should become 0.0
        # quickly since it can never be chosen.
        self.pheromones_stage_one = (
            np.ones(
                (self.problem.jssp.number_of_tasks, len(self.problem.jssp.machines))
            )
            * tau_zero
        )
        self.pheromones_stage_two = (
            np.ones(
                (
                    problem.jssp.number_of_tasks + 1,
                    problem.jssp.number_of_tasks + 1,
                )
            )
            * tau_zero
        )
        self.best_solution: tuple[float, list[int], list[int]] = (1e100, [1], [1])
        self.generation_since_update = 0

    def evaluate(self, path: list[int], machine_assignment: list[int]) -> float:
        """Evaluates a path based on the objective function."""
        if self.objective_function == ObjectiveFunction.MAKESPAN:
            return self.problem.jssp.makespan(path, machine_assignment)
        elif self.objective_function == ObjectiveFunction.MAXIMUM_LATENESS:
            return self.problem.jssp.maximum_lateness(path, machine_assignment)
        else:
            raise ValueError("Objective function not implemented.")

    def draw_transition(self, current: int, valid_moves: list[int]):
        """Make a transistion from the current node to a valid node.

        There is a q_zero probability that the move will be deterministic (i.e. explotation),
        and (1-q_zero) that it will be stochastic (i.e. biased exploration).
        This follows from what they are doing in https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=585892
        """
        # If we are at the starting node, we have uniform distribution for which one to start with
        if current == 0:
            return np.random.choice(valid_moves)

        probabilities = np.zeros(len(valid_moves))
        denominator = 0.0
        for idx, move in enumerate(valid_moves):
            tau_r_s = self.pheromones_stage_two[current, move]
            # Heuristic information, this can be duration or setup times.
            # for setup times we run into the issue of dividing by zero, and having a strong bias towards very low setup times
            eta_r_s = 1 / self.problem.jssp.jobs[move].duration ** self.beta
            numerator = tau_r_s * eta_r_s
            probabilities[idx] = numerator
            denominator += numerator

        # If we are in the q_zero probability we just choose the best one (exploitation)
        if np.random.rand() <= self.q_zero:
            return valid_moves[np.argmax(probabilities)]

        # If we have very low probabilities we just choose randomly
        if denominator <= 1e-6:
            return np.random.choice(valid_moves)

        probabilities = probabilities / denominator
        return np.random.choice(valid_moves, p=probabilities)

    def global_update_pheromones(self):
        """Update the pheromones globally using the best solution found so far.

        Note that, as opposed to many other ACO implementations we are using alpha here
        instead of the usually used rho, to be consistent with the paper of Dorigo & Gambardella."""
        inverse_best_distance = (1.0 / self.best_solution[0]) * self.alpha
        for idx, move in enumerate(self.best_solution[1]):
            # We always (implicitly) start at node 0 or the source node
            # This is why it's also handy that our task counter starts at 1
            if idx == 0:
                self.pheromones_stage_two[0, move] = (
                    1 - self.alpha
                ) * self.pheromones_stage_two[0, move] + inverse_best_distance
            else:
                # Update the pheromones for the transition from the previous node to the current node
                self.pheromones_stage_two[self.best_solution[1][idx - 1], move] = (
                    1 - self.alpha
                ) * self.pheromones_stage_two[
                    self.best_solution[1][idx - 1], move
                ] + inverse_best_distance

        for idx, machine in enumerate(self.best_solution[2]):
            self.pheromones_stage_one[idx, machine] = (
                self.pheromones_stage_one[idx, machine] * (1 - self.alpha)
                + inverse_best_distance
            )
        return

    def local_update_pheromones(self, path: list[int], machine_assignment: list[int]):
        """Updates the pheromones locally using the rho paramater.

        Dorigo and Gambardella tested different functions for delta_tau, but for
        the sake of efficency we will use delta_tau = tau_o for transitions.
        This was also the best performing function in their tests.
        """
        rho_delta_tau = self.rho * self.tau_zero
        for idx, move in enumerate(path):
            if idx == 0:
                self.pheromones_stage_two[0, move] = (
                    1 - self.rho
                ) * self.pheromones_stage_two[0, move] + rho_delta_tau
            else:
                self.pheromones_stage_two[path[idx - 1], move] = (
                    1 - self.rho
                ) * self.pheromones_stage_two[path[idx - 1], move] + rho_delta_tau

        for idx, machine in enumerate(machine_assignment):
            self.pheromones_stage_one[idx, machine] = (
                self.pheromones_stage_one[idx, machine] * (1 - self.rho) + rho_delta_tau
            )
        return

    def assign_machines(self) -> list[int]:
        assignments = list()
        for job in self.problem.jssp.jobs.values():
            available_machines = job.available_machines
            probabilites = np.zeros(len(available_machines))
            if len(available_machines) == 1:
                assignments.append(list(available_machines.keys())[0])
                continue
            for idx, available_machine in enumerate(available_machines):
                tau_r_s = self.pheromones_stage_one[job.task_id - 1, available_machine]
                eta_r_s = 1 / job.duration**self.beta
                probabilites[idx] = tau_r_s * eta_r_s
            probabilites /= np.sum(probabilites)
            chosen_machine = np.random.choice(
                list(available_machines.keys()), p=probabilites
            )
            assignments.append(chosen_machine)
        return assignments

    def run_ant(self) -> tuple[list[int], list[int]]:
        """Guides the ant to a solution and building a path.

        Returns:
            list[int]: the path the ant has taken.
        """
        # We start at the source node, and all the operations that are connected to that
        # one is considered valid moves.
        next_valid_moves: list[int] = [
            n for n in self.problem.jssp.graph.successors("u")
        ]
        path = list()

        # We assign the jobs to machines
        machine_assignment = self.assign_machines()

        # We start at source node also known as node 0
        current = 0
        while len(path) != self.problem.jssp.number_of_tasks:
            # Pick one of the valid moves
            next_move = self.draw_transition(current, next_valid_moves)
            # Add it to the path
            path.append(next_move)
            # Update the current node we are standing on
            current = next_move
            # Get a list with the new valid moves that became available because
            # of our move to the next node.
            new_next_valid_moves = [
                n for n in self.problem.jssp.graph.successors(current) if n != "v"
            ]
            # Remove the current node from the list of valid moves
            next_valid_moves.remove(current)
            next_valid_moves.extend(new_next_valid_moves)

        return path, machine_assignment

    def run_and_update_ant(self) -> tuple[float, list[int], list[int]]:
        """Method for running the ant and updating the pheromones.

        Returns:
            tuple[float, list[int]]: makespan and path of the ant.
        """
        path, machine_assignment = self.run_ant()
        makespan = self.evaluate(path, machine_assignment)
        if makespan < self.best_solution[0]:
            if self.verbose:
                print(f"New best solution found: {makespan}")
            self.best_solution = (makespan, path, machine_assignment)
            self.generation_since_update = 0
        self.local_update_pheromones(path, machine_assignment)
        return (makespan, path, machine_assignment)

    def run(self):
        """Runs the ACO algorithm."""
        for gen in range(self.n_iter):
            self.generation_since_update += 1
            # We keep track of all the solutions found by the ants
            solutions = list()

            for _ in range(self.n_ants):
                res = self.run_and_update_ant()
                # solutions.append(res)

            if self.best_solution[0] == 0:
                break
            # We update the pheromones globally
            self.global_update_pheromones()

            # If we are verbose we print the best makespan every generation
            if self.verbose:
                print(f"Generation {gen}, best makespan: {self.best_solution[0]}")
            elif gen % 20 == 0:
                print(f"Generation {gen}, best makespan: {self.best_solution[0]}")


if __name__ == "__main__":
    # data = parse_data("examples/data_v1_single.xlsx")
    # BKS (makespan): 3825
    # BKS (lateness w/ no earliness bonus): 14600
    # BKS (lateness w/ no earliness bonus + machine hours): 16800
    data = parse_data("examples/data_v1.xlsx")
    machine_aco = FullJobShopProblem.from_data(data)
    machine_aco = assign_machines(machine_aco)
    aco = FullACO(
        machine_aco,
        objective_function=ObjectiveFunction.MAXIMUM_LATENESS,
        verbose=True,
        n_ants=500,
        n_iter=1,
        tau_zero=1.0 / (500.0 * 17985.0),
        seed=2343235,
        beta=1,
        q_zero=0.7,
    )
    aco.run()
    print(f"{aco.best_solution=}")
    schedule = aco.problem.jssp.make_schedule(
        aco.best_solution[1], aco.best_solution[2]
    )
    aco.problem.jssp.visualize_schedule(schedule)
    plt.savefig("example.png")
    # print(jssp.makespan(solve_optimally(jssp.jobs)))
