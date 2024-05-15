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
            np.ones(
                (len(self.problem.jobs), len(self.problem.jobs))
            )
            * tau_zero
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
        self.generation_since_update = 0