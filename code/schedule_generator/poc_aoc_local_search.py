"""Contains the methods for local search"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poc_aoc import Job

def build_precedence_graph(jobs: dict[int, "Job"]) -> nx.DiGraph:
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
    return G


def get_critical_path(graph: nx.DiGraph):
    return nx.dag_longest_path(graph)


def generate_conjunctive_graph(
    graph: nx.DiGraph, jobs: dict[int, "Job"], job_order: list[int]
) -> nx.DiGraph:
    machine_orders = []
    machines = set(j.machine for j in jobs.values())
    edge_list = list()
    for m in machines:
        m_list = list()
        for j in job_order:
            job = jobs[j]
            if job.machine == m:
                if len(m_list) > 0:
                    edge_list.append((m_list[-1], j, {"weight": job.duration}))
                m_list.append(j)
        machine_orders.append(m_list)
    graph.add_edges_from(edge_list)
    return graph


def swap_in_critical_path(
    critical_path,
    job_order: list[int],
    jobs: dict[int, "Job"],
) -> list[int] | None:
    # Find two operations on critical_path on the same machine that we can swap
    potential_swaps: list[tuple[int, int]] = []
    critical_path_idx_mapping = {node: idx for idx, node in enumerate(job_order)}

    for idx, node in enumerate(critical_path[:-1]):
        next_node = critical_path[idx + 1]
        if node in ["u", "v"] or next_node in ["u", "v"]:
            continue
        if (
            jobs[node].machine == jobs[next_node].machine
            and critical_path_idx_mapping[node] + 1
            == critical_path_idx_mapping[next_node]
            and jobs[node].job_id != jobs[next_node].job_id
        ):
            potential_swaps.append((node, next_node))

    # If we have 0 potential swaps we don't have any neighbour solutions
    if len(potential_swaps) == 0:
        return None

    # Take a random swap
    swap = potential_swaps[np.random.randint(len(potential_swaps))]
    # Make the swap
    new_order = job_order.copy()
    new_order[critical_path_idx_mapping[swap[0]]], new_order[critical_path_idx_mapping[swap[1]]] = (
        new_order[critical_path_idx_mapping[swap[1]]],
        new_order[critical_path_idx_mapping[swap[0]]],
    )
    return new_order


# if __name__ == "__main__":
#     jobs_list = [[(0, 3), (1, 2), (2, 2)], [(0, 2), (2, 1), (1, 4)], [(1, 4), (2, 3)]]
#     jobs = generate_job(job_list=jobs_list)
#     G = build_precedence_graph(jobs)
#     job_order = [7, 4, 1, 2, 5, 3, 8, 6]
#     G = generate_conjunctive_graph(G, jobs, job_order)
#     print(calculate_make_span(job_order))
#     cp: list[int] = get_critical_path(G)  # type: ignore
#     new_job_order = swap_in_critical_path(cp, G, job_order, jobs)
#     print(calculate_make_span(new_job_order))
#     nx.draw_networkx(
#         G,
#         pos={
#             "u": (0, 0),
#             "v": (4, 0),
#             1: (1, 1),
#             2: (2, 1),
#             3: (3, 1),
#             4: (1, 0),
#             5: (2, 0),
#             6: (3, 0),
#             7: (1, -1),
#             8: (2, -1),
#         },
#     )
#     plt.show()
