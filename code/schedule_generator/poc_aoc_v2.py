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

import time
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
from pydantic import BaseModel

from poc_aoc_local_search import (
    generate_conjunctive_graph,
    get_critical_path,
    solve_optimally,
)


class Job(BaseModel):
    """Represents a job/task/operation in a job shop problem."""

    duration: int
    machine: int
    dependencies: list[int]
    product_id: int | None  # This will be used later to keep track of the setup times.
    task_id: int
    job_id: int


class JobShopProblem:
    """Represents a job shop problem, with a set of jobs and machines, and a graph of dependencies between the jobs."""

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
        """Builds a graph for the job shop problem with edges as dependencies. It also adds source and sink nodes.

        Note that the weights, i.e. the duration of the tasks are added as the weight _to_ the node.
        That means that if we go from job a to job b the weight for the job b will be on the edge from a to b.
        """
        G = nx.DiGraph()
        G.add_nodes_from(
            ["u", "v"]
            + [(x, {"machine": self.jobs[x].machine}) for x in self.jobs.keys()]
        )
        edges = list()

        # For each job we add an edge, if it lacks dependencies it must start at the source node
        for job_idx, job in self.jobs.items():
            if len(job.dependencies) == 0:
                edges.append(("u", job_idx, {"weight": job.duration}))
            elif len(job.dependencies) == 1:
                edges.append((job.dependencies[0], job_idx, {"weight": job.duration}))
        G.add_edges_from(edges)

        edges = list()
        # The jobs that now have no outgoing edges must end at the sink node, since they do no have
        # any dependencies.
        for node, outdegree in G.out_degree(G.nodes()):
            if outdegree == 0 and node != "v":
                edges.append((node, "v", {"weight": 0}))

        G.add_edges_from(edges)

        return G

    @classmethod
    def from_list(cls, job_list: list[list[tuple[int, int, int | None]]]):
        """Reads a list of jobs and creates a JobShopProblem object from it.

        The jobs list are represented as list of jobs, where each job is a list of tuples.
        Each tuple contains information (machine, duration, product_id (optional))
        """
        jobs: dict[int, Job] = dict()
        # We want to keep track on how many machines we have
        machines = set()

        # We keep track on the amount of tasks we have created to assign them unique task ids
        task_counter = 1
        for idx, job in enumerate(job_list):
            # Since the job list is a list of jobs, we have to add dependencies between the tasks of the same job
            for task_idx, task in enumerate(job):
                task_dict = {}
                if task_idx == 0:
                    task_dict["dependencies"] = []
                else:
                    task_dict["dependencies"] = [task_counter - 1]
                task_dict["duration"] = task[1]
                task_dict["machine"] = task[0]
                # We add the machine to the set of machines
                machines.add(task[0])
                # If we have an assigned product_id we add it to the task
                task_dict["product_id"] = task[2] if len(task) == 3 else None
                task_dict["task_id"] = task_counter
                task_dict["job_id"] = idx
                jobs[task_counter] = Job(**task_dict)
                task_counter += 1

        number_of_jobs = len(job_list)
        number_of_tasks = len(jobs)
        return cls(
            jobs=jobs,
            machines=machines,
            number_of_jobs=number_of_jobs,
            number_of_tasks=number_of_tasks,
        )

    @classmethod
    def from_standard_specification(cls, text: str):
        """Reads a standard job shop problem specification as defined here: http://jobshop.jjvh.nl/explanation.php.

        This is a normal FJSSP with no setup times. The inputed text should not contain any machine and job numbers at the top.
        """
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
        """Returns a schedule where each machine has a list of tasks to perform. The list is a tuple with (task_id, duration, product_id).

        The schedule is created by iterating over the job order and checking wich other relevant tasks there are.
        The relevant tasks are as follows:
        1. The latest task on the same machine, so that we can calculate the earliest start time for the job.
        2. The tasks that the current task is dependent on. Since we cannot start a job before its dependencies are done.

        If the dependencies are not scheduled yet when iteration, that is the job order is not correct, an error is raised.
        """
        # Initialize the schedule with a dummy task for each machine starting and ending at 0
        schedule: dict[int, list[tuple[int, int, int]]] = {
            m: [(-1, 0, 0)] for m in self.machines
        }
        job_schedule = dict()

        # Iterate (in order) over the job order and schedule the tasks
        for task_idx in job_order:
            task: Job = self.jobs[task_idx]
            relevant_tasks = list()
            # The latest job on the same machine is always relevant
            latest_job_on_same_machine = schedule[task.machine][-1]
            relevant_tasks.append(latest_job_on_same_machine)

            # If we have dependencies we need to add them to the relevant tasks
            if len(task.dependencies) > 0:
                for dep in task.dependencies:
                    if dep_task := job_schedule.get(dep):
                        relevant_tasks.append(dep_task)

            # Make sure that we have all dependencies scheduled already
            if len(relevant_tasks) != len(task.dependencies) + 1:
                raise ValueError(
                    f"Dependencies not met for task {task_idx},"
                    f"relevant tasks: {relevant_tasks}.\nSize should be {len(task.dependencies) + 1},"
                    f"but is {len(relevant_tasks)}"
                )

            # The start time is the maximum of the end times of the relevant tasks
            # Since we either have to wait for the machine to be free or the dependencies to be done
            start_time = max([t[2] for t in relevant_tasks])

            # End time with setup time
            end_time = (
                start_time
                + task.duration
                + self.setup_times[latest_job_on_same_machine[0] - 1, task_idx - 1]
            )
            schedule[task.machine].append((task_idx, start_time, end_time))
            job_schedule[task_idx] = (task.machine, start_time, end_time)

        return schedule

    def makespan(self, job_order: list[int]) -> float:
        """Returns the makespan for a given job order"""
        schedule = self.make_schedule(job_order)
        # The makespan is simply the greatest end time of all tasks
        return max([t[-1][2] for t in schedule.values()])


class ACO:
    """Class for the Ant Colony Optimization algorithm for the job shop problem."""

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
        """Initializes the ACO algorithm.

        Implemented according to the paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=585892

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

        # pheromones will be (tasks + 1) x (tasks + 1) matrix, since 0 is the
        # starting node and we start counting tasks from 1, so we need to add 1
        # to the number of tasks. However, (0,0) in pheromones should become 0.0
        # quickly since it can never be chosen.
        self.pheromones = (
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
        denominator = 0.0
        for idx, move in enumerate(valid_moves):
            tau_r_s = self.pheromones[current, move]
            # Heuristic information, this can be duration or setup times.
            # for setup times we run into the issue of dividing by zero, and having a strong bias towards very low setup times
            eta_r_s = 1 / self.problem.jobs[move].duration ** self.beta
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
                self.pheromones[0, move] = (1 - self.alpha) * self.pheromones[
                    0, move
                ] + inverse_best_distance
            else:
                # Update the pheromones for the transition from the previous node to the current node
                self.pheromones[self.best_solution[1][idx - 1], move] = (
                    1 - self.alpha
                ) * self.pheromones[
                    self.best_solution[1][idx - 1], move
                ] + inverse_best_distance
        return

    def local_update_pheromones(self, path: list[int], make_span: float):
        """Updates the pheromones locally using the rho paramater.

        Dorigo and Gambardella tested different functions for delta_tau, but for
        the sake of efficency we will use delta_tau = tau_o for transitions.
        This was also the best performing function in their tests.
        """
        rho_delta_tau = self.rho * self.tau_zero
        for idx, move in enumerate(path):
            if idx == 0:
                self.pheromones[0, move] = (1 - self.rho) * self.pheromones[
                    0, move
                ] + rho_delta_tau
            else:
                self.pheromones[path[idx - 1], move] = (1 - self.rho) * self.pheromones[
                    path[idx - 1], move
                ] + rho_delta_tau
        return

    def run_ant(self) -> list[int]:
        """Guides the ant to a solution and building a path.

        Returns:
            list[int]: the path the ant has taken.
        """
        # We start at the source node, and all the operations that are connected to that
        # one is considered valid moves.
        next_valid_moves: list[int] = [n for n in self.problem.graph.successors("u")]
        path = list()
        # We start at source node also known as node 0
        current = 0
        while len(path) != self.problem.number_of_tasks:
            # Pick one of the valid moves
            next_move = self.draw_transition(current, next_valid_moves)
            # Add it to the path
            path.append(next_move)
            # Update the current node we are standing on
            current = next_move
            # Get a list with the new valid moves that became available because
            # of our move to the next node.
            new_next_valid_moves = [
                n for n in self.problem.graph.successors(current) if n != "v"
            ]
            # Remove the current node from the list of valid moves
            next_valid_moves.remove(current)
            next_valid_moves.extend(new_next_valid_moves)

        return path

    def local_exact_search(self, path: list[int]) -> tuple[list[int], float]:
        """Do a local (exact) search on a part of the critical path.

        We do this by finding a block on the critical path, and then solve it optimally. The
        block size is determined by the left_span, right_span and middle point, which are all
        randomly chosen.

        Args:
            path (list[int]): the path to do the local search on.

        Returns:
            tuple[list[int], float]: tuple with the new path and the makespan.
        """
        # Create a conjunctive graph based on the (implicit) disjunctive graph
        # in other words we connect the nodes that are on the same machine, with
        # directed edges in the order they are in the path.
        conjunctive_graph = generate_conjunctive_graph(
            self.problem.graph.copy(), self.problem.jobs, path # type: ignore
        )
        # We get the critical path from the conjunctive graph, as the longest path
        critical_path: np.ndarray = get_critical_path(conjunctive_graph)  # type: ignore

        left_span = np.random.randint(1, 4)
        right_span = np.random.randint(1, 4)

        # We find a block in the critical path that we want to solve optimally
        critical_path_block_middle = np.random.randint(
            1 + left_span, len(critical_path) - 1 - right_span
        )
        # We need to get the index of the start and end of the block in the actual path
        index_start = path.index(critical_path[critical_path_block_middle - left_span])
        index_end = path.index(critical_path[critical_path_block_middle + right_span])
        job_block = path[index_start : index_end + 1]

        # Solve the block optimally, this is a fairly small problem so it should be done reasonably fast
        new_job_order = solve_optimally(
            {job_id: self.problem.jobs[job_id] for job_id in job_block}
        )

        # Updating the path with the new found optimal solution block
        complete_new_job_order = path.copy()
        complete_new_job_order[index_start : index_end + 1] = new_job_order
        new_time = self.problem.makespan(complete_new_job_order)

        # If we found a better solution we update it
        if new_time < self.best_solution[0]:
            print(f"Local exact search found a better solution: {new_time}")
            self.best_solution = (new_time, complete_new_job_order)
            self.generation_since_update = 0
        return complete_new_job_order, new_time

    def run_and_update_ant(self) -> tuple[float, list[int]]:
        """Method for running the ant and updating the pheromones.

        Returns:
            tuple[float, list[int]]: makespan and path of the ant.
        """
        path = self.run_ant()
        makespan = self.problem.makespan(path)
        if makespan < self.best_solution[0]:
            if self.verbose:
                print(f"New best solution found: {makespan}")
            self.best_solution = (makespan, path)
            self.generation_since_update = 0
        self.local_update_pheromones(path, makespan)
        return (makespan, path)

    def run(self):
        """Runs the ACO algorithm."""
        for gen in range(self.n_iter):
            self.generation_since_update += 1
            # We keep track of all the solutions found by the ants
            solutions = list()

            for _ in range(self.n_ants):
                res = self.run_and_update_ant()
                solutions.append(res)

            # We take the 3 best solutions and do a local exact search on them
            for sol in sorted(solutions, key=lambda x: x[0])[:3]:
                new_path, makespan = self.local_exact_search(sol[1])
                # If we found a better solution we update the pheromones
                # NOTE: this may be a bit too passive, it might perform better
                # if the pheromones are alway updated, since we are traversing a new path
                if makespan < sol[0]:
                    self.local_update_pheromones(new_path, makespan)

            # We update the pheromones globally
            self.global_update_pheromones()

            # If we are verbose we print the best makespan every generation
            if self.verbose:
                print(f"Generation {gen}, best makespan: {self.best_solution[0]}")
            elif gen % 20 == 0:
                print(f"Generation {gen}, best makespan: {self.best_solution[0]}")


if __name__ == "__main__":
    # Optimal solution for la01 is 666
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
    # Optimal solution for ft06 is 55
    ft06_text = """2 1 0 3 1 6 3 7 5 3 4 6
1 8 2 5 4 10 5 10 0 10 3 4
2 5 3 4 5 8 0 9 1 1 4 7
1 5 0 5 2 5 3 3 4 8 5 9
2 9 1 3 4 5 5 4 0 3 3 1
1 3 3 3 5 9 0 10 4 4 2 1"""
    # Optimal solution for la16 is 945
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

    aco = ACO(
        problem,
        n_ants=50,
        n_iter=200,
        verbose=False,
        seed=2234588805,
        beta=2,
        tau_zero=1.0 / (50.0 * 1000.0),
    )
    aco.run()
    print(
        f"Best makespan found: {aco.best_solution[0]}, which is {(aco.best_solution[0] - 945.0) / 945.0:.2%} of the optimal solution for this problem."
    )
    print(f"Path: {aco.best_solution[1]}")
    print(f"Took {time.time() - start_time:.2f} seconds to run.")

    # Plot the pheromones
    plt.imshow(aco.pheromones, cmap="viridis")
    plt.colorbar(label="pheromones Intensity")
    plt.show()
