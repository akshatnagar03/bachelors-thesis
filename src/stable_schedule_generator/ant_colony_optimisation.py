import time
from src.production_orders import parse_data
from src.stable_schedule_generator.main import JobShopProblem, ObjectiveFunction
import numpy as np


class TwoStageACO:
    def __init__(
        self,
        problem: JobShopProblem,
        objective_function: ObjectiveFunction,
        *,
        n_ants: int = 500,
        n_iter: int = 50,
        seed: int = 42,
        rho: float = 0.1,
        alpha: float = 0.1,
        beta: float = 2.0,
        q_zero: float = 0.9,
        tau_zero: float = 1.0,
        verbose: bool = False,
    ) -> None:
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
        self.objective_function = objective_function

        # pheromones will be (tasks + 1) x (tasks + 1) matrix, since 0 is the
        # starting node and we start counting tasks from 1, so we need to add 1
        # to the number of tasks. However, (0,0) in pheromones should become 0.0
        # quickly since it can never be chosen.
        self.pheromones_stage_one = (
            np.ones((len(self.problem.jobs), len(self.problem.machines))) * tau_zero
        )
        self.pheromones_stage_two = (
            np.ones(
                (
                    len(self.problem.jobs) + 1,
                    len(self.problem.jobs) + 1,
                )
            )
            * tau_zero
        )
        self.best_solution: tuple[float, list[int], list[int]] = (1e100, [1], [1])

    def evaluate(self, machine_assignment: list[int], path: list[int]) -> float:
        """Evaluates the path and machine assignment."""
        schedule = self.problem.make_schedule(path, machine_assignment)
        if self.objective_function == ObjectiveFunction.MAKESPAN:
            return self.problem.makespan(schedule)
        elif self.objective_function == ObjectiveFunction.TARDINESS:
            return self.problem.tardiness(schedule)
        elif self.objective_function == ObjectiveFunction.TOTAL_SETUP_TIME:
            return self.problem.total_setup_time(schedule)
        else:
            raise ValueError(
                f"Objective function {self.objective_function} not supported."
            )

    def draw_transition(
        self, current: int, valid_moves: list[int], available_machines: list[int]
    ) -> int:
        if current == -1:
            return np.random.choice(valid_moves)

        probabilities = np.zeros(len(valid_moves))
        denominator = 0.0
        for idx, move in enumerate(valid_moves):
            tau_r_s = self.pheromones_stage_two[current + 1, move + 1]
            eta_r_s = (
                1.0
                / self.problem.jobs[move].available_machines[available_machines[move]]
            )
            numerator = tau_r_s * eta_r_s**self.beta
            denominator += numerator
            probabilities[idx] = numerator

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
        # Update stage one pheromones
        for idx, machine in enumerate(self.best_solution[1]):
            self.pheromones_stage_one[idx, machine] = (
                self.pheromones_stage_one[idx, machine] * (1 - self.alpha)
                + inverse_best_distance
            )

        # Update stage two pheromones
        for idx, move in enumerate(self.best_solution[2]):
            self.pheromones_stage_two[idx + 1, move + 1] = (
                self.pheromones_stage_two[idx + 1, move + 1] * (1 - self.alpha)
                + inverse_best_distance
            )

    def assign_machines(self) -> list[int]:
        assignment = list()
        for idx, job in enumerate(self.problem.jobs):
            available_machines = list(job.available_machines.keys())
            if len(available_machines) == 1:
                assignment.append(available_machines[0])
                continue
            probabilities = np.zeros(len(available_machines))
            denominator = 0.0
            for idx, machine in enumerate(available_machines):
                tau_r_s = self.pheromones_stage_one[idx, machine]
                eta_r_s = 1.0 / job.available_machines[machine]
                numerator = tau_r_s * eta_r_s**self.beta
                probabilities[idx] = numerator
                denominator += numerator

            if denominator <= 1e-6:
                assignment.append(np.random.choice(available_machines))
                continue
            probabilities = probabilities / denominator
            assignment.append(np.random.choice(available_machines, p=probabilities))
        return assignment

    def run_ant(self) -> tuple[list[int], list[int]]:
        path = list()
        machine_assignment = self.assign_machines()
        current = -1
        valid_moves = [
            i
            for i in range(len(self.problem.jobs))
            if len(self.problem.jobs[i].dependencies) == 0
        ]
        while len(path) != len(self.problem.jobs):
            next_move = self.draw_transition(current, valid_moves, machine_assignment)
            path.append(next_move)
            # Update pheromones
            self.pheromones_stage_two[current + 1, next_move + 1] = (
                self.pheromones_stage_two[current + 1, next_move + 1] * (1 - self.rho)
                + self.tau_zero * self.rho
            )
            current = next_move
            valid_moves.remove(next_move)
            new_valid_moves = [
                n for n in self.problem.graph.successors(current) if n >= 0
            ]
            valid_moves.extend(new_valid_moves)

        return machine_assignment, path

    def run_and_update_ant(self) -> tuple[float, list[int], list[int]]:
        machine_assignment, path = self.run_ant()
        objective_value = self.evaluate(
            machine_assignment=machine_assignment, path=path
        )
        if objective_value < self.best_solution[0]:
            self.best_solution = (objective_value, machine_assignment, path)
            if self.verbose:
                print(f"New best solution found: {self.best_solution[0]}")
        return objective_value, machine_assignment, path

    def run(self):
        try:
            for gen in range(self.n_iter):
                for _ in range(self.n_ants):
                    self.run_and_update_ant()
                if self.verbose:
                    print(
                        f"Generation {gen}, best objective value: {self.best_solution[0]}"
                    )
                elif gen % 10 == 0:
                    print(
                        f"Generation {gen}, best objective value: {self.best_solution[0]}"
                    )
                self.global_update_pheromones()
        except KeyboardInterrupt:
            print("Got stop signal, stopping early...")
            print(f"{self.best_solution=}")


if __name__ == "__main__":
    data = parse_data("examples/data_v1.xlsx")
    jssp = JobShopProblem.from_data(data)
    start_time = time.time()
    aco = TwoStageACO(jssp, ObjectiveFunction.MAKESPAN, verbose=True, n_iter=10)
    aco.run()
    print(aco.best_solution)
    print(f"Time taken: {time.time() - start_time}")
    aco.problem.visualize_schedule(
        aco.problem.make_schedule(aco.best_solution[2], aco.best_solution[1])
    )
