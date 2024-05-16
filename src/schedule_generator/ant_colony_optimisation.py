import time
from src.production_orders import parse_data
from src.schedule_generator.main import JobShopProblem, ObjectiveFunction
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
                    len(self.problem.machines),
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
        elif self.objective_function == ObjectiveFunction.CUSTOM_OBJECTIVE:
            return self.problem.custom_objective(schedule)
        else:
            raise ValueError(
                f"Objective function {self.objective_function} not supported."
            )

    def assign_machines(self) -> dict[int, set[int]]:
        assignment: dict[int, set[int]] = {machine: set() for machine in range(len(self.problem.machines))}
        for idx, job in enumerate(self.problem.jobs):
            available_machines = list(job.available_machines.keys())
            if len(available_machines) == 1:
                assignment[available_machines[0]].add(idx)
                continue
            probabilities = np.zeros(len(available_machines))
            denominator = 0.0
            for machine_idx, machine in enumerate(available_machines):
                tau_r_s = self.pheromones_stage_one[idx, machine]
                eta_r_s = 1.0 / job.available_machines[machine]
                numerator = tau_r_s * eta_r_s**self.beta
                probabilities[machine_idx] = numerator
                denominator += numerator

            if denominator <= 1e-6:
                chosen_machine = np.random.choice(available_machines)
                assignment[chosen_machine].add(idx)
                continue
            probabilities = probabilities / denominator
            chosen_machine = np.random.choice(available_machines, p=probabilities)
            assignment[chosen_machine].add(idx)
        return assignment

    def global_update_pheromones(self):...

    def draw_job_to_schedule(self, jobs_to_schedule: set[int], last: int, machine: int) -> int:
        jobs_to_schedule_list = list(jobs_to_schedule)
        if last == -1:
            return np.random.choice(jobs_to_schedule_list)
        
        probabilites = np.zeros(len(jobs_to_schedule_list))
        denominator = 0.0
        for idx, job in enumerate(jobs_to_schedule_list):
            tau_r_s = self.pheromones_stage_two[last, job, machine]
            eta_r_s = 1.0 / self.problem.jobs[job].available_machines[machine]
            numerator = tau_r_s * eta_r_s**self.beta
            probabilites[idx] = numerator
            denominator += numerator

        if np.random.rand() <= self.q_zero:
            return jobs_to_schedule_list[np.argmax(probabilites)]
        if denominator <= 1e-6:
            return np.random.choice(jobs_to_schedule_list)
        probabilites = probabilites / denominator
        return np.random.choice(jobs_to_schedule_list, p=probabilites)

    def run_and_update_ant(self):
        machine_assignment = self.assign_machines() 
        print(machine_assignment)
        schedules: list[list[int]] = [list() for _ in range(len(self.problem.machines))]
        for machine in range(len(self.problem.machines)):
            schedules[machine].append(-1)
            jobs_assigned = set()
            for _ in machine_assignment[machine]:
                job_idx = self.draw_job_to_schedule(jobs_to_schedule=machine_assignment[machine].difference(jobs_assigned),last=schedules[machine][-1], machine=machine)
                schedules[machine].append(job_idx)
                jobs_assigned.add(job_idx)

        print(self.problem.make_schedule_from_parallel(schedules))



    def run(self):
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


if __name__ == "__main__":
    data = parse_data("examples/data_v1_single.xlsx")
    jssp = JobShopProblem.from_data(data)
    start_time = time.time()
    aco = TwoStageACO(jssp, ObjectiveFunction.TARDINESS, verbose=True, n_iter=1,n_ants=1, tau_zero=1.0 / (500*16000.0), q_zero=0.85)
    aco.run()
    print(aco.best_solution)
    print(f"Time taken: {time.time() - start_time}")
    # aco.problem.visualize_schedule(
    #     aco.problem.make_schedule(aco.best_solution[2], aco.best_solution[1])
    #     , "examples/schedule_tardiness_100_new.png"
    # )
