import numpy as np
from src.schedule_generator.poc_aco_v2 import ACO, Job, JobShopProblem
from src.schedule_generator.poc_aco_machine_assignment import ACOMachine
from src.production_orders import Data, Workstation, parse_data


def from_assigned_machine_to_jssp(aco_machine: ACOMachine) -> JobShopProblem:
    jobs: dict[int, Job] = dict()

    for sub_job in aco_machine.sub_jobs:
        if sub_job.machine == -1:
            raise ValueError("Sub-job has not been assigned a machine.")
        jobs[sub_job.task_id] = sub_job

    jssp = JobShopProblem(jobs, machines=set(aco_machine.machine_key.values()),number_of_jobs=len(jobs)//2,number_of_tasks=len(jobs))

    setup_times = np.zeros((len(jobs), len(jobs)))
    job_keys = list(jobs.keys())
    reverse_machine_key = {v: k for k, v in aco_machine.machine_key.items()}
    for j1 in job_keys:
        for j2 in job_keys:
            j1_job = jobs[j1]
            j2_job = jobs[j2]
            
            # If the jobs are from the same product, there are no setup times
            if j1_job.product_id == j2_job.product_id:
                continue
            # If the jobs are on different machines, there are no setup times
            if j1_job.machine != j2_job.machine:
                continue

            machine: Workstation = [station for station in aco_machine.problem.workstations if station.name == reverse_machine_key[j1_job.machine]][0]
            # If the jobs have different tastes, there is a setup time for both bottling and mixing
            if j1_job.station_settings["taste"] != j2_job.station_settings["taste"]:
                setup_times[j1, j2] += machine.minutes_changeover_time_taste
            
            # If the jobs are on a bottling line and have different bottle sizes, there is a setup time
            if j1_job.station_settings["bottle_size"] != j2_job.station_settings["bottle_size"] and "bottling" in machine.name.lower():
                setup_times[j1, j2] += machine.minutes_changeover_time_bottle_size

    jssp.set_setup_times(setup_times)
    return jssp
             
        


def assign_machines(aco_machine: ACOMachine) -> ACOMachine:
    for sub_job in aco_machine.sub_jobs:
        to_assign = list(sub_job.available_machines.keys())[0]
        sub_job.assign_machine(to_assign)
    return aco_machine


if __name__ == "__main__":
    # data = parse_data("examples/data_v1_single.xlsx")
    data = parse_data("examples/data_v1.xlsx")
    machine_aco = ACOMachine.from_data(data)
    machine_aco = assign_machines(machine_aco)
    jssp = from_assigned_machine_to_jssp(machine_aco)
    aco = ACO(jssp, verbose=True)
    aco.run()