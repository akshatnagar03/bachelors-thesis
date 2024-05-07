"""Handles the assignment of the machines for the jobs

There will be a N x M matrix, where M is number of machines, N is total number of jobs.
This matrix has the pheremone levels that job i is assigned to machine k (i,k). The
heuristic value (visibility) will either be the inverse of the processing time, or something with setuptime, maybe dynamic?
"""

from typing import Any, Self
from src.production_orders import Data, parse_data
from src.schedule_generator.poc_aco_v2 import Job


class ACOMachine:
    def __init__(self, problem: Data, sub_jobs: list[Job], machine_key: dict[str, int]):
        self.problem = problem
        self.sub_jobs = sub_jobs
        self.machine_key = machine_key

    @classmethod
    def from_data(cls, data: Data) -> Self:
        """This creates a new ACOMachine assignment problem from the given data.
        It will create the sub-jobs, so that they can fit the machines (i.e. batches of ) in the data."""
        sub_jobs = []
        job_counter = 0
        mixing_lines = [
            station.name
            for station in data.workstations
            if station.name.lower().startswith("mixing")
        ]
        bottling_lines = [
            station.name
            for station in data.workstations
            if station.name.lower().startswith("bottling")
        ]
        mixing_line_max_amount: int = min(
            data.workstation_df["max_units_per_run"]
            .loc[data.workstation_df["name"].isin(mixing_lines)]
            .values
        )
        machine_key = {
            station_name: i for i, station_name in enumerate(mixing_lines + bottling_lines)
        }
        for order in data.production_orders:
            # create sub-jobs for each order
            bill_of_materials = data.bill_of_materials_df.loc[
                data.bill_of_materials_df["parent_id"] == order.product_id
            ]
            hf_product = bill_of_materials["component_id"].values[0]
            bottle_size = bill_of_materials["component_quantity"].values[0]
            station_settings = {"taste": hf_product, "bottle_size": bottle_size}
            # HACK: Assume that there are 2 steps for each order for now
            number_of_batches = int(
                (order.amount * bottle_size) // mixing_line_max_amount
            )
            for i in range(number_of_batches):
                if i == number_of_batches - 1:
                    size_of_job = int(
                        order.amount * bottle_size % mixing_line_max_amount
                    )
                    if size_of_job == 0:
                        size_of_job = mixing_line_max_amount
                else:
                    size_of_job = mixing_line_max_amount
                # first consider the mixing line
                sub_jobs.append(
                    Job(
                        duration=-1,
                        machine=-1,
                        dependencies=list(),
                        product_id=order.product_id,
                        days_till_delivery=order.days_till_delivery,
                        available_machines={
                            machine_key[station_name]: data.workstation_df.loc[
                                data.workstation_df["name"] == station_name
                            ]["minutes_per_run"].values[0]
                            for station_name in mixing_lines
                        },
                        station_settings=station_settings,
                        amount=size_of_job,
                        production_order_nr=order.production_order_nr,
                        task_id=job_counter,
                        job_id=order.production_order_nr
                    )
                )
                job_counter += 1

                # then consider the bottling line
                sub_jobs.append(
                    Job(
                        duration=-1,
                        machine=-1,
                        dependencies=[job_counter - 1],
                        product_id=order.product_id,
                        days_till_delivery=order.days_till_delivery,
                        available_machines={
                            machine_key[station_name]: data.workstation_df.loc[
                                data.workstation_df["name"] == station_name
                            ]["minutes_per_run"].values[0]
                            * size_of_job
                            for station_name in bottling_lines
                        },
                        station_settings=station_settings,
                        amount=size_of_job,
                        production_order_nr=order.production_order_nr,
                        task_id=job_counter,
                        job_id=order.production_order_nr
                    )
                )
                job_counter += 1
        return cls(data, sub_jobs, machine_key)

    def __str__(self) -> str:
        return f"ACOMachine(problem={self.problem}, sub_jobs={self.sub_jobs})"


if __name__ == "__main__":
    data = parse_data("examples/data_v1.xlsx")
    ACOMachine.from_data(data)
