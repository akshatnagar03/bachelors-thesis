import time
from src.production_orders import parse_data
from src.schedule_generator.main import JobShopProblem, ObjectiveFunction
from src.schedule_generator.numba_numpy_functions import select_random_item, nb_set_seed
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
        with_stock_schedule: bool = False,
        with_local_search: bool = True,
        local_search_iterations: int = 20,
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
        nb_set_seed(seed)
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
            np.ones((len(self.problem.jobs), len(self.problem.machines)))  # * tau_zero
        )
        self.pheromones_stage_two = (
            np.ones(
                (
                    len(self.problem.jobs) + 1,
                    len(self.problem.jobs) + 1,
                    len(self.problem.machines),
                )
            )
            # * tau_zero
        )
        self.best_solution: tuple[float, np.ndarray] = (1e100, np.zeros((1, 1)))
        self.with_stock_schedule = with_stock_schedule
        self.with_local_search = with_local_search
        self.local_search_iterations = local_search_iterations
        self.generation_since_last_update = 0

    def evaluate(self, parallel_schedule: np.ndarray) -> float:
        """Evaluates the path and machine assignment."""
        if self.with_stock_schedule:
            schedule = self.problem.make_schedule_from_parallel_with_stock(
                parallel_schedule
            )
        else:
            schedule = self.problem.make_schedule_from_parallel(parallel_schedule)
        if self.objective_function == ObjectiveFunction.MAKESPAN:
            return self.problem.makespan(schedule)
        elif self.objective_function == ObjectiveFunction.TARDINESS:
            return self.problem.tardiness(schedule)
        elif self.objective_function == ObjectiveFunction.TOTAL_SETUP_TIME:
            return self.problem.total_setup_time(schedule)
        elif self.objective_function == ObjectiveFunction.CUSTOM_OBJECTIVE:
            return self.problem.custom_objective(schedule)
        elif self.objective_function == ObjectiveFunction.BOOLEAN_TARDINESS:
            return self.problem.boolean_tardiness(schedule)
        else:
            raise ValueError(
                f"Objective function {self.objective_function} not supported."
            )

    def assign_machines(self) -> dict[int, set[int]]:
        assignment: dict[int, set[int]] = {
            machine: set() for machine in range(len(self.problem.machines))
        }
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
                chosen_machine = select_random_item(available_machines)
                assignment[chosen_machine].add(idx)
                continue
            probabilities = probabilities / denominator
            chosen_machine = select_random_item(
                available_machines, probabilities=probabilities
            )
            assignment[chosen_machine].add(idx)
        return assignment

    def global_update_pheromones(self):
        inverse_best_value = (1.0 / self.best_solution[0]) * self.tau_zero
        for m_idx, order in enumerate(self.best_solution[1]):
            for idx, job_idx in enumerate(order):
                if idx == 0 or job_idx == -1:
                    continue
                # Update stage one
                self.pheromones_stage_one[job_idx, m_idx] = (
                    self.pheromones_stage_one[job_idx, m_idx] * (1 - self.alpha)
                    + self.alpha * inverse_best_value
                )
                # Update stage two
                last_job_idx = order[idx - 1]
                self.pheromones_stage_two[last_job_idx, job_idx, m_idx] = (
                    self.pheromones_stage_two[last_job_idx, job_idx, m_idx]
                    * (1 - self.alpha)
                    + self.alpha * inverse_best_value
                )

    def local_update_pheromones(self, schedule: np.ndarray):
        for machine in range(len(self.problem.machines)):
            for idx, job_idx in enumerate(schedule[machine]):
                if idx == 0 or job_idx == -1:
                    continue
                if job_idx == -2:
                    break
                # Update stage one
                self.pheromones_stage_one[job_idx, machine] = (
                    self.pheromones_stage_one[job_idx, machine] * (1 - self.rho)
                    + self.rho * 0.4  # * self.tau_zero
                )

                # Update stage two
                last_job_idx = schedule[machine][idx - 1]
                self.pheromones_stage_two[last_job_idx, job_idx, machine] = (
                    self.pheromones_stage_two[last_job_idx, job_idx, machine]
                    * (1 - self.rho)
                    + self.rho * 0.4  # * self.tau_zero
                )

    def draw_job_to_schedule(
        self, jobs_to_schedule: set[int], last: int, machine: int
    ) -> int:
        jobs_to_schedule_list = list(jobs_to_schedule)
        if last == -1:
            return select_random_item(jobs_to_schedule_list)

        probabilites = np.zeros(len(jobs_to_schedule_list))
        denominator = 0.0
        last_job_prod_order = self.problem.jobs[last].production_order_nr
        for idx, job in enumerate(jobs_to_schedule_list):
            if self.problem.jobs[job].production_order_nr == last_job_prod_order:
                coef = 0.5
            else:
                coef = 1.0
            tau_r_s = self.pheromones_stage_two[last, job, machine]
            eta_r_s = 1.0 / (
                (
                    1
                    # self.problem.jobs[job].available_machines[machine]
                    + self.problem.setup_times[last, job]
                )
                * coef
            )
            numerator = tau_r_s * eta_r_s**self.beta
            probabilites[idx] = numerator
            denominator += numerator

        if np.random.rand() <= self.q_zero:
            return jobs_to_schedule_list[np.argmax(probabilites)]
        if denominator <= 1e-7:
            return select_random_item(jobs_to_schedule_list)
        probabilites = probabilites / denominator
        return select_random_item(jobs_to_schedule_list, probabilities=probabilites)

    def local_search(
        self,
        schedule: np.ndarray,
        machine_assignment: dict[int, set[int]],
        schedule_objective_value: float,
    ):
        new_schedule = schedule  # .copy()
        for _ in range(self.local_search_iterations):
            x = np.random.rand()
            operation = (0, 0, 0)
            if x < 0.5:
                # Swap two jobs on the same machine
                machine = np.random.randint(len(self.problem.machines))
                number_of_jobs_on_machine = len(machine_assignment[machine])
                if number_of_jobs_on_machine < 2:
                    continue
                job1_idx = np.random.randint(1, number_of_jobs_on_machine)
                job2_idx = np.random.randint(1, number_of_jobs_on_machine)
                new_schedule[machine][job1_idx], new_schedule[machine][job2_idx] = (
                    new_schedule[machine][job2_idx],
                    new_schedule[machine][job1_idx],
                )
                operation = (machine, job1_idx, job2_idx)
            objective_value = self.evaluate(new_schedule)
            if objective_value < schedule_objective_value:
                schedule = new_schedule
            else:
                machine, job1_idx, job2_idx = operation
                new_schedule[machine][job1_idx], new_schedule[machine][job2_idx] = (
                    new_schedule[machine][job2_idx],
                    new_schedule[machine][job1_idx],
                )

    def run_ant(self) -> tuple[np.ndarray, dict[int, set[int]]]:
        machine_assignment = self.assign_machines()
        # schedules: list[list[int]] = [list() for _ in range(len(self.problem.machines))]
        schedules = (
            np.ones(
                (len(self.problem.machines), len(self.problem.jobs)), dtype=np.int32
            )
            * -2
        )
        for machine in range(len(self.problem.machines)):
            schedules[machine, 0] = -1
            jobs_assigned = set()
            for i in range(len(machine_assignment[machine])):
                job_idx = self.draw_job_to_schedule(
                    jobs_to_schedule=machine_assignment[machine].difference(
                        jobs_assigned
                    ),
                    last=schedules[machine][-1],
                    machine=machine,
                )
                schedules[machine, i + 1] = job_idx
                jobs_assigned.add(job_idx)
        return schedules, machine_assignment

    def run_and_update_ant(self):
        schedule, machine_assignment = self.run_ant()
        objective_value = self.evaluate(schedule)
        if self.with_local_search:
            self.local_search(schedule, machine_assignment, objective_value)
        self.local_update_pheromones(schedule)
        if objective_value <= self.best_solution[0]:
            if objective_value == 0:
                raise KeyboardInterrupt
            self.best_solution = (objective_value, schedule)
            self.generation_since_last_update = 0
            if self.verbose:
                print(f"New best solution: {self.best_solution[0]}")

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
            self.generation_since_last_update += 1
            if self.generation_since_last_update == 100:
                print("Resetting pheromones...")
                self.pheromones_stage_one *= 0
                self.pheromones_stage_one += 1
                self.pheromones_stage_two *= 0
                self.pheromones_stage_two += 1
                self.generation_since_last_update = 0


if __name__ == "__main__":
    data = parse_data("examples/data_v1.xlsx")
    jssp = JobShopProblem.from_data(data)
    aco = TwoStageACO(
        jssp,
        ObjectiveFunction.CUSTOM_OBJECTIVE,
        verbose=True,
        n_iter=1,
        n_ants=1,
        tau_zero=1.0 / (0.1),
        q_zero=0.85,
        with_stock_schedule=False,
        seed=65490,
        with_local_search=False,
        local_search_iterations=20,
    )
    # start_time = time.time()
    aco.run()
    # print(f"Time taken: {time.time() - start_time}")
    # print(aco.best_solution)
    # sc = aco.problem.make_schedule_from_parallel(aco.best_solution[1])
    # aco.problem.visualize_schedule(
    #     sc
    # )
    # aco.problem.visualize_schedule(
    #     aco.problem.make_schedule_from_parallel(aco.best_solution[1])
    # )
