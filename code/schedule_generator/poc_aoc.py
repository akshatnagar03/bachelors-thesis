"""Proof of Concept for Ant Colony Optimization Algorithm.
This PoC is initially trying to generate a schedule, based on the constraints and a vector
with consequitive tasks.

The idea is that a two-stage Ant Colony Optimization will happen.
--- Stage 1 ---
In the first stage each job will be assigned to one of the machines they could be assigned to.
This could be either through ACO, but it might make more sense to use GA here.

--- Stage 2 ---
For this stage the true ACO will take place. We will let the ants run on the graph and
put out phermones and follow them. However we will also keep track of which tasks ("sub jobs") the ant has
already assigned. There is also evidence that suggests that using local search could
significantly improve the result of this stage.
"""

from functools import cache
from itertools import product
import time
from typing import Any, Callable
import networkx as nx
from pydantic import BaseModel, ConfigDict
import numpy as np
import matplotlib.pyplot as plt

from poc_aoc_local_search import solve_optimally, swap_in_critical_path, get_critical_path, generate_conjunctive_graph


class Job(BaseModel):
    duration: int
    dependencies: list[int]
    machine: int
    job_id: int
    task_id: int


# Jobs in this list consists of tuples of (machine, duration)
jobs_list = [[(0, 3), (1, 2), (2, 2)], [(0, 2), (2, 1), (1, 4)], [(1, 4), (2, 3)]]


def read_jssp(text: str):
    list_of_jobs = list()
    rows = text.split("\n")
    for row in rows:
        numbers = row.split()
        machine_numbers = numbers[::2]
        machine_time = numbers[1::2]
        list_of_jobs.append(
            [(int(m), int(t)) for m, t in zip(machine_numbers, machine_time)]
        )
    return list_of_jobs


# ft10 found at http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop1.txt
# best known solution: 930
# text = """0 29 1 78 2  9 3 36 4 49 5 11 6 62 7 56 8 44 9 21
#  0 43 2 90 4 75 9 11 3 69 1 28 6 46 5 46 7 72 8 30
#  1 91 0 85 3 39 2 74 8 90 5 10 7 12 6 89 9 45 4 33
#  1 81 2 95 0 71 4 99 6  9 8 52 7 85 3 98 9 22 5 43
#  2 14 0  6 1 22 5 61 3 26 4 69 8 21 7 49 9 72 6 53
#  2 84 1  2 5 52 3 95 8 48 9 72 0 47 6 65 4  6 7 25
#  1 46 0 37 3 61 2 13 6 32 5 21 9 32 8 89 7 30 4 55
#  2 31 0 86 1 46 5 74 4 32 6 88 8 19 9 48 7 36 3 79
#  0 76 1 69 3 76 5 51 2 85 9 11 6 40 7 89 4 26 8 74
#  1 85 0 13 2 61 6  7 8 64 9 76 5 47 3 52 4 90 7 45"""

# la01 best known solution: 666
text = """1 21 0 53 4 95 3 55 2 34
 0 21 3 52 4 16 2 26 1 71
 3 39 4 98 1 42 2 31 0 12
 1 77 0 55 4 79 2 66 3 77
 0 83 3 34 2 64 1 19 4 37
 1 54 2 43 4 79 0 92 3 62
 3 69 4 77 1 87 2 87 0 93
 2 38 0 60 1 41 3 24 4 83
 3 17 1 49 4 25 0 44 2 98
 4 77 3 79 2 43 1 75 0 96"""

# text = """2  1  0  3  1  6  3  7  5  3  4  6
#  1  8  2  5  4 10  5 10  0 10  3  4
#  2  5  3  4  5  8  0  9  1  1  4  7
#  1  5  0  5  2  5  3  3  4  8  5  9
#  2  9  1  3  4  5  5  4  0  3  3  1
#  1  3  3  3  5  9  0 10  4  4  2  1"""

jobs_list = read_jssp(text)

# Dict that will contain all the jobs and their relevant information
# The key is the job number, and the value is a dict with the following keys:
# - duration: The duration of the task
# - dependencies: A list of job numbers that this job depends on (that needs to be completed before this job can start)
# - machine: The machine that this job should be executed on
# Example:
# jobs = {
#     1: {"duration": 3, "dependencies": [], "machine": 1},
#     2: {"duration": 5, "dependencies": [], "machine": 1},
#     3: {"duration": 5, "dependencies": [1], "machine": 2},
#     4: {"duration": 2, "dependencies": [2], "machine": 2},
# }

def generate_job_list(job_list: list[list[tuple[int, int]]]) -> dict[int, Job]:
    jobs: dict[int, Job] = {}
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
            task_dict["job_id"] = idx
            task_dict["task_id"] = task_counter
            jobs[task_counter] = Job(**task_dict)
            task_counter += 1

    return jobs

jobs = generate_job_list(jobs_list)


# The job_order list contains the order in which the jobs should be executed.
# If there are two jobs in the list say [1, 2] that are executed on different machines,
# then job 2 could start before job 1 starts. However if job 2 is dependent on job 1
# then job 2 is started only once job 1 is completed.
# job_order = [3, 0, 6, 4, 5, 7, 1, 2]


def make_schedule(job_order) -> dict[int, list[tuple[int, int, int]]]:
    # Contains all the machines that are utilized
    machines = set([job.machine for job in jobs.values()])

    # Schedule contains the machines, and the tasks that are assigned to them
    # The list consists of tuples of (task, start_time, end_time)
    schedule: dict[int, list[tuple[int, int, int]]] = {
        machine: [(-1, 0, 0)] for machine in machines
    }

    for job in job_order:
        machine: int = jobs[job].machine
        dependencies: list[int] = jobs[job].dependencies
        duration: int = jobs[job].duration
        relevant_tasks = []
        # The last task on the machine is always relevant
        relevant_tasks.append(schedule[machine][-1])

        # If we have dependencies, we need to add them to the relevant task
        if len(dependencies) > 0:
            for m in schedule.keys():
                for task in schedule[m]:
                    # Find the task in the schedule that corresponds to the dependency
                    if task[0] in dependencies:
                        relevant_tasks.append(task)

        # If we cannot find all the dependencies that means that the job order is invalid
        # since job_order should guarantee that all dependencies are met
        # We add 1 to the length, since we always have the last task on the machine
        if len(relevant_tasks) != len(dependencies) + 1:
            raise Exception(f"Dependencies not met for job {job} on machine {machine}\n{job_order}")

        # Calculate the start time of the job as the maximum end_time in the relevant tasks
        start_time = max([task[2] for task in relevant_tasks])

        # Add the task to the schedule with the start time and end times calculated
        schedule[machine].append((job, start_time, start_time + duration))

    return schedule

def list_to_tuple(function: Callable) -> Any:
    """Custom decorator function, to convert list to a tuple."""

    def wrapper(*args, **kwargs) -> Any:
        args = tuple(tuple(x) if isinstance(x, list) else x for x in args)
        kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in kwargs.items()}
        result = function(*args, **kwargs)
        result = tuple(result) if isinstance(result, list) else result
        return result

    return wrapper

@list_to_tuple
@cache
def calculate_make_span(job_order):
    """For each job we will try to start it as early as possible
    We can do this by checking the dependencies and the last task on the machine
    If the dependencies are not met, we will raise an error, since that should not happen
    in a valid job order (precedence constraint should hold for job_order)
    Once we have the relevant tasks, we can calculate the start time of the job by
    taking the maximum end time of the relevant tasks"""

    schedule = make_schedule(job_order=job_order)

    # Print schedule and L_max
    # print(schedule)
    max_end_time = 0
    for machine in schedule.keys():
        for task in schedule[machine]:
            if max_end_time < task[2]:
                max_end_time = task[2]
    # print(f"Max end time: {max_end_time}")
    return max([task[2] for task in [m[-1] for m in schedule.values()]])


# Build a graph from the jobs dict
G = nx.DiGraph()
# We have a source and sink node
G.add_nodes_from(["u", "v"])

# Add all the task nodes
nodes = [(x, {}) for x in jobs]
G.add_nodes_from(nodes)

# We need to create edges between all the nodes
edges = []
for job_idx, job in jobs.items():
    # If we have an independent job (i.e. it can start whenever) it gets assigned to the source node
    if len(job.dependencies) == 0:
        edges.append(("u", job_idx, {"weight": 0}))
    # If we have a dependency that one will have an edge between each other (directed)
    # With the duration as weight
    if len(job.dependencies) == 1:
        edges.append(
            (
                job.dependencies[0],
                job_idx,
                {"weight": jobs[job.dependencies[0]].duration},
            )
        )

G.add_edges_from(edges)


# Add the sink node edges for the nodes that only have a degree of 1
# We ignore the sink and source nodes
edges = [
    (node, "v", {"weight": jobs[node].duration})
    for node, outdegree in G.out_degree(G.nodes())
    if outdegree == 0 and node != "v"
]


G.add_edges_from(edges)

# import matplotlib.pyplot as plt
# pos = nx.planar_layout(G)
# nx.draw_networkx(G, pos=pos, with_labels=True)
# plt.show()


class JobShopSchedulingProblem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    jobs: dict[int, Job]
    graph: nx.DiGraph


class ACO:
    """Class for Ant Colony Optimization. Equivialent to stage 2."""

    def __init__(
        self,
        problem: JobShopSchedulingProblem,
        *,
        rho: float = 0.1,
        tau: float = 1.0,
        tau_min: float = 0.02,
        n_ants: int = 60,
        n_iter: int = 10,
        alpha: float = 1.0,
        phi: float = 1.0,
        beta: float = 1.0,
        seed: int = 42,
        number_of_jobs: int | None = None,
        with_local_search: bool = False,
        local_iterations: int = 10,
        with_unfirom_search: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize the Ant Colony Optimization algorithm for job shop scheduling.

        Args:
            problem (JobShopSchedulingProblem): The problem instance to be solved.
            rho (float, optional): Pheromone evaporation rate. Defaults to 0.1.
            tau (float, optional): Initial pheromone value. Defaults to 10.0.
            n_ants (int, optional): Number of ants. Defaults to 60.
            n_iter (int, optional): Number of iterations. Defaults to 10.
            alpha (float, optional): Importance of pheromone. Defaults to 1.0.
            beta (float, optional): Importance of heuristic information. Defaults to 1.0.
            seed (int, optional): Sets the seed for np.random
        """
        self.jobs = problem.jobs
        self.graph = problem.graph
        self.rho = rho
        self.phi = phi
        self.tau_max = tau
        self.tau_min = tau_min
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        # We have a (jobs + 1, jobs) size matrix, we add 1 since we start at job -1
        self.phermones = np.ones((len(self.jobs) + 1, len(self.jobs) + 1)) * tau
        np.random.seed(seed)
        self.with_local_search = with_local_search
        self.local_iterations = local_iterations
        self.with_uniform_search = with_unfirom_search
        self.best_solution = (1e100, [1])
        self.verbose = verbose
        self.generations_since_last_improvement = 0
        if number_of_jobs:
            self.number_of_jobs = number_of_jobs
        else:
            self.number_of_jobs = len(self.jobs)

    def run(self):
        if self.verbose:
            print(
                f"Running ACO with:\n{self.rho=}\n{self.phi=}\n{self.alpha=}\n{self.beta=}\n{self.n_ants=}\n{self.n_iter=}"
            )
        last_best = self.best_solution[0]
        for gen in range(self.n_iter):
            self.uniform_threashold = np.tanh(self.generations_since_last_improvement / 1000)
            if gen % 20 == 0 and not self.verbose:
                print(f"Generation: {gen}, best found: {self.best_solution[0]}")
            solutions = []
            for ant in range(self.n_ants):
                ants_job_order = self.generate_solution()
                ants_objective_time_value = self.evaluate(ants_job_order)
                solutions.append((ants_objective_time_value, ants_job_order))

            if self.verbose:
                c_max_list = [s[0] for s in solutions]
                print(
                    f"Gen: {gen}, Overall best: {self.best_solution[0]}, min: {np.min(c_max_list)}, average: {np.average(c_max_list):.1f}, max: {np.max(c_max_list)}, diversity: {len(set([s[0] for s in solutions]))/self.n_ants:.2f}"
                )

            if last_best == self.best_solution[0]:
                self.generations_since_last_improvement += 1
            else:
                last_best = self.best_solution[0]
                self.generations_since_last_improvement = 0

            self.local_exact_search(solutions=solutions)

            if self.with_local_search:
                self.local_search(solutions=solutions)
            if self.with_uniform_search:
                self.uniform_search()

            self.update_phermones(solutions=solutions)

    def uniform_search(self):
        best_time = self.best_solution[0]
        number_of_ants = self.generations_since_last_improvement
        for ant in range(number_of_ants):
            ants_job_order = self.generate_solution(uniform=True)
            self.evaluate(ants_job_order)
        if self.best_solution[0] != best_time:
            print(f"Uniform search found a better solution: {self.best_solution[0]}")


    def local_search(self, solutions: list[tuple[int, list[int]]]):
        iter_best_solution = solutions[np.argmin([s[0] for s in solutions])]
        s_o_time = iter_best_solution[0]
        s_o = iter_best_solution[1]
        s_k = s_o
        s_k_time = s_o_time
        a = 0.9
        for i in range(self.local_iterations):
            conjunctive_graph = generate_conjunctive_graph(self.graph.copy(), self.jobs, s_k)
            critical_path = get_critical_path(conjunctive_graph)
            s_c = swap_in_critical_path(critical_path, s_o, self.jobs)
            if not s_c:
                continue
            s_c_time = self.evaluate(s_c)
            if s_o_time < s_c_time:
                s_o = s_c
                s_o_time = s_c_time
                s_k = s_c
                s_k_time = s_c_time
            elif s_c_time < s_k_time:
                s_k = s_c
                s_k_time = s_c_time
            else:
                p_s_k_s_c = np.exp((s_k_time - s_c_time) / (a))
                if np.random.rand() < p_s_k_s_c:
                    s_k = s_c
                    s_k_time = s_c_time
            a = a * 0.9
        if s_o_time < self.best_solution[0]:
            print(f"Local search found a better solution: {s_o_time}")
            self.best_solution = (s_o_time, s_o)
        if s_o_time < iter_best_solution[0]:
            print(f"Local search found a better solution (generationally): {s_o_time}")

    def local_exact_search(self, solutions: list[tuple[int, list[int]]]):
        # take best solution
        # find a critical path block and solve it optimally
        iter_best_job = solutions[np.argmin([s[0] for s in solutions])]
        conjunctive_graph = generate_conjunctive_graph(self.graph.copy(), self.jobs, iter_best_job[1]) # type: ignore
        critical_path: np.ndarray = get_critical_path(conjunctive_graph) # type: ignore
        critical_path_block_middle = np.random.randint(2, len(critical_path) - 2)
        index_start = iter_best_job[1].index(critical_path[critical_path_block_middle - 1])
        index_end = iter_best_job[1].index(critical_path[critical_path_block_middle + 1])
        job_block = iter_best_job[1][index_start:index_end + 1]
        # solve it optimally

        new_job_order = solve_optimally({job_id: self.jobs[job_id] for job_id in job_block})
        complete_new_job_order = iter_best_job[1].copy()
        complete_new_job_order[index_start:index_end + 1] = new_job_order
        new_time = calculate_make_span(complete_new_job_order)
        if new_time < self.best_solution[0]:
            print(f"Local exact search found a better solution: {new_time}")
            self.best_solution = (new_time, complete_new_job_order)


    def generate_solution(self, *, uniform: bool = False) -> list[int]:
        next_valid_moves: list[int] = [n for n in self.graph.successors("u")]
        path = list()
        current = 0
        next_move_uniform = uniform
        while len(path) < len(self.jobs):
            if current == 0:
                next_move_uniform = True
            elif not uniform:
                next_move_uniform = False

            # returns one tuple with the decided move
            selected_move = self._select_move(
                current=current, valid_moves=next_valid_moves, uniform=next_move_uniform
            )
            # update the graph progress
            path.append(selected_move)
            current = selected_move
            index_of_done_move = next_valid_moves.index(selected_move)
            if next_move := self._get_next_valid_move(selected_move):
                next_valid_moves[index_of_done_move] = next_move
            else:
                next_valid_moves.pop(index_of_done_move)
        return path

    def _get_next_valid_move(self, move: int) -> int | None:
        next_moves = [n for n in self.graph.successors(move) if n != "v"]
        if len(next_moves) > 0:
            return next_moves[0]
        return None

    def _select_move(
        self, valid_moves: list[int], current: int, uniform: bool = False
    ) -> int:
        probabilities = np.zeros(len(valid_moves))
        for idx, move in enumerate(valid_moves):
            tau = self.phermones[current, move] ** self.alpha
            eta = (1.0 / self.jobs[move].duration) ** self.beta
            probability = tau * eta
            probabilities[idx] = probability

        probabilities = probabilities / sum(probabilities)

        if np.random.rand() < self.uniform_threashold or uniform:
            idx = np.random.choice([i for i in range(len(valid_moves))])
        else:
            idx = np.random.choice([i for i in range(len(valid_moves))], p=probabilities)

        return valid_moves[idx]

    def evaluate(self, job_order: list[int]) -> int:
        make_span = calculate_make_span(job_order=job_order)
        if make_span < self.best_solution[0]:
            self.best_solution = (make_span, job_order)
        return make_span

    def update_phermones(self, solutions: list[tuple[int, list[int]]]):
        phermone_update = self._get_best_path(solutions)
        phermone_update = np.sqrt(phermone_update)
        # We might be able to achive the same result by first multiplying with (1 - self.rho)
        # and then add the path matrix with the rewards
        for i, j in product(
            range(self.phermones.shape[0]), range(self.phermones.shape[1])
        ):
            self.phermones[i, j] = min(max((1 - self.rho) * self.phermones[i, j] + phermone_update[i, j], self.tau_min), self.tau_max)

    def _get_best_path(self, solutions: list[tuple[int, list[int]]]) -> np.ndarray:
        phermone_update = np.zeros((self.phermones.shape[0], self.phermones.shape[1]))
        delta_tau_best = (1.0 / self.best_solution[0]) * self.phi 
        # delta_tau_best = self.tau_min * 3
        for idx, val in enumerate(self.best_solution[1]):
            if idx == 0:
                phermone_update[0, val] += delta_tau_best 
            else:
                phermone_update[self.best_solution[1][idx - 1], val] += delta_tau_best 

        return phermone_update
        for sol_make_span, ant_path in solutions:
            delta_tau_ant = (1.0 / sol_make_span) 
            # delta_tau_ant = self.tau_min * 3
            for idx, val in enumerate(ant_path):
                if idx == 0:
                    phermone_update[0, val] += delta_tau_ant
                # Since we will encounter the next task of the same job must more often than any other job, we will be conservative on
                # assigning phermones to that path
                elif (prev := ant_path[idx-1]) != val - 1:
                    phermone_update[prev, val] += delta_tau_ant
                else:
                    phermone_update[prev, val] += delta_tau_ant / (len(self.jobs) / 2)

        return phermone_update

    def visualize_best(self):
        schedule = make_schedule(self.best_solution[1])

        fig, ax = plt.subplots(figsize=(10, 6))

        # Define a color map for machines
        cmap = plt.get_cmap("tab10")

        column_major_job_id = (
            np.arange(1, self.number_of_jobs * 5 + 1)
            .reshape((self.number_of_jobs, 5), order="C")
            .flatten(order="F")
        )
        # Plotting tasks for each machine
        for i, (machine, schedule) in enumerate(schedule.items()):
            for task in schedule:
                job_id, start_time, end_time = task
                if job_id == -1:
                    continue
                ax.plot(
                    [start_time, end_time],
                    [i + 1, i + 1],
                    linewidth=10,
                    solid_capstyle="butt",
                    alpha=0.8,
                    color=cmap(jobs[job_id].job_id % 10),
                    label=self.jobs[job_id].job_id,
                )
                label = str(column_major_job_id[job_id - 1]),
                label = job_id
                ax.text(
                    (start_time + end_time) / 2,
                    i + 1,
                    label,
                    va="center",
                    ha="right",
                    fontsize=12,
                )

        handles, labels = ax.get_legend_handles_labels()
        unique = [
            (h, l)
            for i, (h, l) in enumerate(zip(handles, labels))
            if l not in labels[:i]
        ]
        ax.legend(*zip(*unique), loc="lower center", ncols=self.number_of_jobs)

        plt.xlabel("Time")
        plt.ylabel("Machines")
        plt.title("Gantt Chart")
        plt.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    problem = JobShopSchedulingProblem(jobs=jobs, graph=G)
    aco = ACO(
        problem,
        n_iter=500,
        n_ants=50,
        tau=1.0,
        tau_min=1e-4,
        number_of_jobs=3,
        phi=1,
        rho=0.1,
        beta=1,
        alpha=1,
        local_iterations=20,
        with_local_search=False,
        with_unfirom_search=False,
        verbose=False,
        seed=232435
    )
    # aco.best_solution = (704, [16, 46, 21, 31, 1, 36, 26, 11, 32, 6, 2, 3, 17, 41, 27, 47, 33, 28, 48, 18, 37, 7, 34, 42, 22, 43, 8, 23, 4, 35, 5, 29, 24, 25, 12, 30, 9, 38, 19, 49, 39, 44, 13, 45, 10, 20, 50, 14, 40, 15])
    aco.run()
    print(aco.best_solution)
    print(np.round(aco.phermones, 1))
    # # aco.visualize_best()
    plt.imshow(aco.phermones, cmap='viridis')
    plt.colorbar(label='Pheromone Intensity')
    plt.show()