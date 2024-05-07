"""Contains the methods for local search"""

import networkx as nx
from typing import TYPE_CHECKING
import pyomo.environ as pyo

if TYPE_CHECKING:
    from src.schedule_generator.poc_aco_v2 import Job


def get_critical_path(graph: nx.DiGraph):
    return nx.dag_longest_path(graph)


def generate_conjunctive_graph(
    graph: nx.DiGraph, jobs: dict[int, "Job"], job_order: list[int]
) -> nx.DiGraph:
    """Generate a conjunctive graph from a precedence graph, with the help of a job order."""
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


def solve_optimally(jobs: dict[int, "Job"]) -> list[int]:
    """Solve the scheduling problem optimally (FJSSP, without setup times)

    Args:
        jobs (dict[int, Job]): jobs that needs to be scheduled

    Returns:
        list[int]: job order of the optimal solution
    """
    model = pyo.ConcreteModel()
    model.tasks = pyo.Set(initialize=jobs.keys())

    model.start = pyo.Var(model.tasks, domain=pyo.NonNegativeReals)
    model.c_max = pyo.Var(domain=pyo.NonNegativeReals)
    # This will be 1 if task1 is before task2
    model.precedence_var = pyo.Var(
        [
            (task1, task2)
            for task1 in model.tasks
            for task2 in model.tasks
            if jobs[task1].machine == jobs[task2].machine and task1 != task2 # type: ignore
        ],
        domain=pyo.Binary,
    )

    model.obj = pyo.Objective(expr=model.c_max, sense=pyo.minimize)

    # Could be done simpler with only constraints for the last jobs
    def c_max_rule(m, task):
        return m.start[task] + jobs[task].duration <= m.c_max

    model.c_max_constraint = pyo.Constraint(model.tasks, rule=c_max_rule)

    def precedence_rule(m, task):
        if len(jobs[task].dependencies) == 0:
            return pyo.Constraint.Skip
        return (
            pyo.quicksum(
                m.start[dep] + jobs[dep].duration
                for dep in jobs[task].dependencies
                if dep in m.tasks
            )
            <= m.start[task]
        )

    model.precedence_constraint = pyo.Constraint(model.tasks, rule=precedence_rule)

    def machine_rule_one(m, task1, task2):
        if (task1, task2) in m.precedence_var:
            return m.start[task1] + jobs[task1].duration <= m.start[task2] + 10e4 * (
                0 + m.precedence_var[task1, task2]
            )
        else:
            return pyo.Constraint.Skip

    model.machine_constraint_one = pyo.Constraint(
        model.tasks, model.tasks, rule=machine_rule_one
    )

    def machine_rule_two(m, task1, task2):
        if (task1, task2) in m.precedence_var:
            return m.precedence_var[task1, task2] + m.precedence_var[task2, task1] <= 1
        else:
            return pyo.Constraint.Skip

    model.machine_constraint_two = pyo.Constraint(
        model.tasks, model.tasks, rule=machine_rule_two
    )

    try:
        # pyo.SolverFactory("cbc").solve(model)
        pyo.SolverFactory(
            "cplex", executable=r"B:\Programs\cplex\cplex\bin\x64_win64\cplex.exe"
        ).solve(model, tee=True)
    except Exception as e:
        print(f"Warning: Could not solve the problem optimally.\n {e}")
        return []

    tasks = {task: model.start[task]() for task in model.tasks}  # type: ignore

    return [task for task, _ in sorted(tasks.items(), key=lambda x: x[1])]  # type: ignore
