"""Proof of Concept for Ant Colony Optimization Algorithm.
This PoC is initially trying to generate a schedule, based on the constraints and a vector 
with consequitive tasks.

The idea is that a two-stage Ant Colony Optimization will happen.
--- Stage 1 ---
In the first stage each job will be assigned to one of the machines they could be assigned to.
This could be either through ACO, but it might make more sense to use GA here.

--- Stage 2 ---
For this stage the true ACO will take place. We will let the ants run on the graph and 
put out phermones. However we will also keep track of which tasks ("sub jobs") the ant has
already assigned. There is also evidence that suggests that using local search could 
significantly improve the result of this stage.
"""
import networkx as nx
from pydantic import BaseModel

class Job(BaseModel):
    duration: int
    dependencies: list[int]
    machine: int

# Jobs in this list consists of tuples of (machine, duration)
jobs_list = [[(0, 3), (1, 2), (2, 2)], [(0, 2), (2, 1), (1, 4)], [(1, 4), (2, 3)]]

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
jobs: dict[int, Job] = {}

task_counter = 0
for job in jobs_list:
    for task_idx, task in enumerate(job):
        task_dict = {}
        if task_idx == 0:
            task_dict["dependencies"] = []
        else:
            task_dict["dependencies"] = [task_counter - 1]
        task_dict["duration"] = task[1]
        task_dict["machine"] = task[0]
        jobs[task_counter] = Job(**task_dict)
        task_counter += 1


# The job_order list contains the order in which the jobs should be executed.
# If there are two jobs in the list say [1, 2] that are executed on different machines,
# then job 2 could start before job 1 starts. However if job 2 is dependent on job 1
# then job 2 is started only once job 1 is completed.
job_order = [3, 0, 6, 4, 5, 7, 1, 2]

# Contains all the machines that are utalized
machines = set([job.machine for job in jobs.values()])

# Schedule contains the machines, and the tasks that are assigned to them
# The list consists of tuples of (task, start_time, end_time)
schedule: dict[int, list[tuple[int, int, int]]] = {
    machine: [(-1, 0, 0)] for machine in machines
}

# For each job we will try to start it as early as possible
# We can do this by checking the dependencies and the last task on the machine
# If the dependencies are not met, we will raise an error, since that should not happen
# in a valid job order (precedence constraint should hold for job_order)
# Once we have the relevant tasks, we can calculate the start time of the job by
# taking the maximum end time of the relevant tasks
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
        raise Exception(f"Dependencies not met for job {job} on machine {machine}")

    # Calculate the start time of the job as the maximum end_time in the relevant tasks
    start_time = max([task[2] for task in relevant_tasks])

    # Add the task to the schedule with the start time and end times calculated
    schedule[machine].append((job, start_time, start_time + duration))

# Print schedule and L_max
print(schedule)
max_end_time = 0
for machine in schedule.keys():
    for task in schedule[machine]:
        if max_end_time < task[2]:
            max_end_time = task[2]
print(f"Max end time: {max_end_time}")

# Build a graph from the jobs dict
G = nx.DiGraph()
# We have a source and sink node
G.add_nodes_from(["u", "v"])

# Add all the task nodes
nodes = [(x,{}) for x in jobs]
G.add_nodes_from(nodes)

# We need to create edges between all the nodes
edges = []
for (job_idx, job) in jobs.items():
    # If we have an independent job (i.e. it can start whenever) it gets assigned to the source node
    if len(job.dependencies) == 0:
        edges.append(("u", job_idx, {"weight": 0}))
    # If we have a dependency that one will have an edge between each other (directed)
    # With the duration as weight
    if len(job.dependencies) == 1:
        edges.append((job.dependencies[0], job_idx, {"weight": jobs[job.dependencies[0]].duration}))
    
G.add_edges_from(edges)

edges = []

# Add the sink node edges for the nodes that only have a degree of 1
# We ignore the sink and source nodes
for node in G.nodes:
    if node != "u" and node != "v" and G.degree[node] == 1: # type: ignore
        edges.append((node, "v", {"weight": jobs[node].duration})) 


G.add_edges_from(edges)

# pos = nx.planar_layout(G)
# nx.draw_networkx(G, pos=pos, with_labels=True)
# plt.show()