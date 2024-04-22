import time
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
from pydantic import BaseModel, ConfigDict

from poc_aoc_local_search import (
    generate_conjunctive_graph,
    get_critical_path,
    solve_optimally,
)


class Job(BaseModel):
    duration: int
    machine: int
    dependencies: list[int]
    product_id: int | None
    task_id: int
    job_id: int


class JobShopProblem:
    def __init__(
        self,
        jobs: dict[int, Job],
        machines: set[int],
        number_of_jobs: int,
        number_of_tasks: int,
    ):
        self.jobs = jobs
        self.machines = machines
        self.number_of_jobs = number_of_jobs
        self.number_of_tasks = number_of_tasks
        self.setup_times = np.zeros((number_of_tasks, number_of_tasks))
        self.graph = self._build_graph()

    def _build_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        G.add_nodes_from(
            ["u", "v"]
            + [(x, {"machine": self.jobs[x].machine}) for x in self.jobs.keys()]
        )
        edges = list()
        for job_idx, job in self.jobs.items():
            if len(job.dependencies) == 0:
                edges.append(("u", job_idx, {"weight": job.duration}))
            elif len(job.dependencies) == 1:
                edges.append((job.dependencies[0], job_idx, {"weight": job.duration}))
        G.add_edges_from(edges)

        edges = list()
        for node, outdegree in G.out_degree(G.nodes()):
            if outdegree == 0 and node != "v":
                edges.append((node, "v", {"weight": 0}))

        G.add_edges_from(edges)

        return G

    @classmethod
    def from_list(cls, job_list: list[list[tuple[int, int, int | None]]]):
        """Reads a list of jobs and creates a JobShopProblem object from it.

        The jobs list are represented as list of jobs, where each job is a list of tuples.
        Each tuple contains information (machine, duration, product_id (optional))"""
        jobs: dict[int, Job] = {}
        machines = set()
        task_counter = 1
        for idx, job in enumerate(job_list):
            for task_idx, task in enumerate(job):
                task_dict = {}
                if task_idx == 0:
                    task_dict["dependencies"] = []
                else:
                    task_dict["dependencies"] = [task_counter - 1]
                task_dict["duration"] = task[1]
                task_dict["machine"] = task[0]
                machines.add(task[0])
                task_dict["product_id"] = task[2] if len(task) == 3 else None
                task_dict["task_id"] = task_counter
                task_dict["job_id"] = idx
                jobs[task_counter] = Job(**task_dict)
                task_counter += 1

        number_of_jobs = len(job_list)
        number_of_tasks = task_counter - 1
        return cls(
            jobs=jobs,
            machines=machines,
            number_of_jobs=number_of_jobs,
            number_of_tasks=number_of_tasks,
        )

    @classmethod
    def from_standard_specification(cls, text: str):
        """Reads a standard job shop problem specification as defined here: http://jobshop.jjvh.nl/explanation.php"""
        list_of_jobs = list()
        raw_jobs = text.split("\n")
        for r in raw_jobs:
            numbers = r.split()
            machine_numbers = [int(i) for i in numbers[::2]]
            duration = [int(i) for i in numbers[1::2]]
            job = list(zip(machine_numbers, duration))
            list_of_jobs.append(job)

        return cls.from_list(list_of_jobs)

    def make_schedule(
        self, job_order: list[int]
    ) -> dict[int, list[tuple[int, int, int]]]:
        """Returns a schedule where each machine has a list of tasks to perform.
        The list is a tuple with (task_id, duration, product_id)"""
        schedule: dict[int, list[tuple[int, int, int]]] = {
            m: [(-1, 0, 0)] for m in self.machines
        }
        for task_idx in job_order:
            task: Job = self.jobs[task_idx]
            relevant_tasks = list()
            latest_job_on_same_machine = schedule[task.machine][-1]
            relevant_tasks.append(latest_job_on_same_machine)

            if len(task.dependencies) > 0:
                for m in self.machines:
                    for t in schedule[m]:
                        if t[0] == task.dependencies[0]:
                            relevant_tasks.append(t)

            if len(relevant_tasks) != len(task.dependencies) + 1:
                raise ValueError(
                    f"Dependencies not met for task {task_idx},"
                    f"relevant tasks: {relevant_tasks}.\nSize should be {len(task.dependencies) + 1},"
                    f"but is {len(relevant_tasks)}"
                )

            start_time = max([t[2] for t in relevant_tasks])
            end_time = (
                start_time
                + task.duration
                + self.setup_times[latest_job_on_same_machine[0] - 1, task_idx - 1]
            )
            schedule[task.machine].append((task_idx, start_time, end_time))

        return schedule

    def makespan(self, job_order: list[int]) -> float:
        """Returns the makespan for a given job order"""
        schedule = self.make_schedule(job_order)
        return max([t[-1][2] for t in schedule.values()])


class ACO:
    def __init__(
        self,
        problem: JobShopProblem,
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
    ):
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

        # Phermones will be (tasks + 1) x (tasks + 1) matrix, since 0 is the
        # starting node and we start counting tasks from 1, so we need to add 1
        # to the number of tasks. However, (0,0) in phermones should become 0.0
        # quickly since it can never be chosen.
        self.phermones = (
            np.ones((problem.number_of_tasks + 1, problem.number_of_tasks + 1))
            * tau_zero
        )
        self.best_solution: tuple[float, list[int]] = (1e100, [1])
        self.generation_since_update = 0

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
        for idx, move in enumerate(valid_moves):
            tau_r_s = self.phermones[current, move]
            eta_r_s = 1 / self.problem.jobs[move].duration ** self.beta
            numerator = tau_r_s * eta_r_s
            probabilities[idx] = numerator

        if np.random.rand() <= self.q_zero:
            return valid_moves[np.argmax(probabilities)]

        denominator = np.sum(probabilities)
        if denominator == 0:
            return np.random.choice(valid_moves)

        probabilities = probabilities / denominator
        return np.random.choice(valid_moves, p=probabilities)

    def global_update_phermones(self):
        """Update the phermones globally using the best solution found so far.

        Note that, as opposed to many other ACO implementations we are using alpha here
        instead of the usually used rho, to be consistent with the paper of Dorigo & Gambardella."""
        inverse_best_distance = 1.0 / self.best_solution[0]
        for idx, move in enumerate(self.best_solution[1]):
            if idx == 0:
                self.phermones[0, move] = (1 - self.alpha) * self.phermones[
                    0, move
                ] + self.alpha * inverse_best_distance
            else:
                self.phermones[self.best_solution[1][idx - 1], move] = (
                    1 - self.alpha
                ) * self.phermones[
                    self.best_solution[1][idx - 1], move
                ] + self.alpha * inverse_best_distance
        return

    def local_update_phermones(self, path: list[int], make_span: float):
        """Updates the phermones locally using the rho paramater.

        Dorigo and Gambardella tested different functions for delta_tau, but for
        the sake of efficency we will use delta_tau = tau_o for transitions.
        This was also the best performing function in their tests.
        """
        rho_delta_tau = self.rho * self.tau_zero
        for idx, move in enumerate(path):
            if idx == 0:
                self.phermones[0, move] = (1 - self.rho) * self.phermones[
                    0, move
                ] + rho_delta_tau
            else:
                self.phermones[path[idx - 1], move] = (1 - self.rho) * self.phermones[
                    path[idx - 1], move
                ] + rho_delta_tau
        return

    def run_ant(self):
        next_valid_moves: list[int] = [n for n in self.problem.graph.successors("u")]
        path = list()
        current = 0
        while len(path) != self.problem.number_of_tasks:
            next_move = self.draw_transition(current, next_valid_moves)
            path.append(next_move)
            current = next_move
            new_next_valid_moves = [
                n for n in self.problem.graph.successors(current) if n != "v"
            ]
            next_valid_moves.remove(current)
            next_valid_moves.extend(new_next_valid_moves)

        return path

    def local_search(self, path: list[int]):
        conjunctive_graph = generate_conjunctive_graph(
            self.problem.graph.copy(), self.problem.jobs, path
        )
        critical_path: np.ndarray = get_critical_path(conjunctive_graph) # type: ignore
        left_span = np.random.randint(1, 4)
        right_span = np.random.randint(1, 4)
        critical_path_block_middle = np.random.randint(
            1 + left_span, len(critical_path) - 1 - right_span
        )
        index_start = path.index(critical_path[critical_path_block_middle - left_span])
        index_end = path.index(critical_path[critical_path_block_middle + right_span])
        job_block = path[index_start : index_end + 1]
        # solve it optimally

        new_job_order = solve_optimally(
            {job_id: self.problem.jobs[job_id] for job_id in job_block}
        )
        complete_new_job_order = path.copy()
        complete_new_job_order[index_start : index_end + 1] = new_job_order
        new_time = self.problem.makespan(complete_new_job_order)
        if new_time < self.best_solution[0]:
            print(f"Local exact search found a better solution: {new_time}")
            self.best_solution = (new_time, complete_new_job_order)
            self.generation_since_update = 0
        return complete_new_job_order, new_time

    def run(self):
        for gen in range(self.n_iter):
            self.generation_since_update += 1
            solutions = list()
            for _ in range(self.n_ants):
                path = self.run_ant()
                makespan = self.problem.makespan(path)
                solutions.append((makespan, path))
                if makespan < self.best_solution[0]:
                    if self.verbose:
                        print(f"New best solution found: {makespan}")
                    self.best_solution = (makespan, path)
                    self.generation_since_update = 0
                self.local_update_phermones(path, makespan)

            for sol in sorted(solutions, key=lambda x: x[0])[:3]:
                new_path, makespan = self.local_search(sol[1])
                if makespan < sol[0]:
                    self.local_update_phermones(new_path, makespan)




            self.global_update_phermones()
            if self.verbose:
                print(f"Generation {gen}, best makespan: {self.best_solution[0]}")
            elif gen % 20 == 0:
                print(f"Generation {gen}, best makespan: {self.best_solution[0]}")


if __name__ == "__main__":
    la01_text = """1 21 0 53 4 95 3 55 2 34
 0 21 3 52 4 16 2 26 1 71
 3 39 4 98 1 42 2 31 0 12
 1 77 0 55 4 79 2 66 3 77
 0 83 3 34 2 64 1 19 4 37
 1 54 2 43 4 79 0 92 3 62
 3 69 4 77 1 87 2 87 0 93
 2 38 0 60 1 41 3 24 4 83
 3 17 1 49 4 25 0 44 2 98
 4 77 3 79 2 43 1 75 0 96"""
    ft06_text = """2 1 0 3 1 6 3 7 5 3 4 6
1 8 2 5 4 10 5 10 0 10 3 4
2 5 3 4 5 8 0 9 1 1 4 7
1 5 0 5 2 5 3 3 4 8 5 9
2 9 1 3 4 5 5 4 0 3 3 1
1 3 3 3 5 9 0 10 4 4 2 1"""
    la16_text = """1 21 6 71 9 16 8 52 7 26 2 34 0 53 4 21 3 55 5 95
4 55 2 31 5 98 9 79 0 12 7 66 1	42	8	77	6	77	3	39
3	34	2	64	8	62	1	19	4	92	9	79	7	43	6	54	0	83	5	37
1	87	3	69	2	87	7	38	8	24	9	83	6	41	0	93	5	77	4	60
2	98	0	44	5	25	6	75	7	43	1	49	4	96	9	77	3	17	8	79
2	35	3	76	5	28	9	10	4	61	6	9	0	95	8	35	1	7	7	95
3	16	2	59	0	46	1	91	9	43	8	50	6	52	5	59	4	28	7	27
1	45	0	87	3	41	4	20	6	54	9	43	8	14	5	9	2	39	7	71
4	33	2	37	8	66	5	33	3	26	7	8	1	28	6	89	9	42	0	78
8	69	9	81	2	94	4	96	3	27	0	69	7	45	6	78	1	74	5	84

"""
    # problem = JobShopProblem.from_standard_specification(la01_text)
    # problem = JobShopProblem.from_standard_specification(ft06_text)
    problem = JobShopProblem.from_standard_specification(la16_text)
    # problem = JobShopProblem.from_list([[(0, 3, None), (1, 2, None), (2, 2, None)], [(0, 2, None), (2, 1, None), (1, 4, None)], [(1, 4, None), (2, 3, None)]])

    start_time = time.time()

    aco = ACO(problem, n_ants=50, n_iter=4000, verbose=False, seed=2345005, beta=1)
    aco.run()
    print(
        f"Best makespan found: {aco.best_solution[0]}, which is {(aco.best_solution[0] - 945.0) / 945.0:.2%} of the optimal solution for this problem."
    )
    print(f"Path: {aco.best_solution[1]}")
    print(f"Took {time.time() - start_time:.2f} seconds to run.")

    # Plot the phermones
    plt.imshow(aco.phermones, cmap="viridis")
    plt.colorbar(label="Pheromone Intensity")
    plt.show()
