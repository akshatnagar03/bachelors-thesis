from enum import Enum
from typing import Any, Self
import numpy as np
from pydantic import BaseModel
from src.production_orders import Data, parse_data, Product, BillOfMaterial


class Job(BaseModel):
    available_machines: list[int]
    dependencies: list[int]
    production_order_nr: str
    station_settings: dict[str, Any] = dict()
    amount: int = 1


class Machine(BaseModel):
    name: str
    machine_id: int
    start_time: int
    end_time: int
    allow_preemption: bool = False
    max_units_per_run: int = 1


class JobShopProblem:
    def __init__(self, data: Data, jobs: list[Job], machines: list[Machine]) -> None:
        self.data: Data = data
        self.jobs: list[Job] = jobs
        self.machines: list[Machine] = machines
        self.setup_times: np.ndarray = np.zeros((len(jobs), len(jobs)))

    @classmethod
    def from_data(cls, data: Data) -> Self:
        """Generate a JobShopProblem from the data provided according to the excel sheet.

        Args:
            data (Data): The data that is parsed from the excel sheet.

        Returns:
            Self: The JobShopProblem object.
        """
        sub_jobs: list[Job] = list()
        machines: list[Machine] = [
            Machine(
                name=m.name,
                machine_id=idx,
                start_time=m.starts_at.hour * 60 + m.starts_at.minute,
                end_time=m.stops_at.hour * 60 + m.stops_at.minute,
                allow_preemption=(m.name.lower().startswith("bottling")),
                max_units_per_run=m.max_units_per_run,
            )
            for idx, m in enumerate(data.workstations)
        ]
        for order in data.production_orders:
            # We collect all the products we need to produce
            # 1. We get the final product from the products table
            # 2. We look in the bill of materials to see if that product has any sub products that are needed
            # 3. If it has, we add the sub product from the products table and repeat step 2.
            products = list()
            curent_product = order.product_id
            while curent_product is not None:
                product = data.products[curent_product]
                bill_of_materials = data.bill_of_materials.get(curent_product, None)
                products.append((product, bill_of_materials))
                if bill_of_materials:
                    curent_product = bill_of_materials.component_id
                else:
                    curent_product = None
            # We reverse the products list, because the products have to be produced in sequential order
            products: list[tuple[Product, BillOfMaterial | None]] = products[::-1]
            # We split the job into batches of the same product
            product_info = list()
            # NOTE: amount is different from the amount in the job, since we calcualte the
            # amount based on what unit the machine can process
            for idx, (prod, _) in enumerate(products):
                prod_info = dict()
                if idx + 1 < len(products):
                    bom = products[idx + 1][1]
                else:
                    bom = None
                amount = order.amount * bom.component_quantity if bom else order.amount
                prod_info["amount"] = amount
                # HACK: we are shortening the name by 1 since it says bottle and not bottling, which the machine name is
                prod_info["machines"] = [
                    m.machine_id
                    for m in machines
                    if m.name.lower().startswith(prod.workstation_type[:-1])
                ]
                # HACK: the batch sizes are not calculated per machine, so if two machines
                # that process the same job has different ones, we will take the one that has
                # the lowest capacity to calculate the batch size
                min_max_units_per_run = min(
                    [machines[m].max_units_per_run for m in prod_info["machines"]]
                )
                prod_info["batches"] = int(amount // min_max_units_per_run)
                prod_info["batches_amount"] = min_max_units_per_run
                remainder = int(amount % min_max_units_per_run)
                prod_info["batches_remainder"] = (
                    remainder if remainder > 0 else min_max_units_per_run
                )
                product_info.append(prod_info)
            # We calculate the correct batch size, or in other words the job that has
            # the smallest batch amount size, since that will be the bottle neck.
            # However, we ignore if the jobs can be run independently, since we will just make that into
            # one batch
            batch_info = min(
                product_info,
                key=lambda x: x["batches_amount"] if x["batches_amount"] > 1 else 10e4,
            )
            for i in range(batch_info["batches"]):
                for idx, prod in enumerate(product_info):
                    dependencies = list()
                    if idx > 0:
                        dependencies.append(len(sub_jobs) - 1)

                    amount = prod["amount"] // batch_info["batches"]
                    if i == batch_info["batches"] - 1:
                        amount = prod["amount"] % batch_info["batches"]
                        if amount == 0:
                            amount = prod["amount"] // batch_info["batches"]

                    sub_jobs.append(
                        Job(
                            available_machines=prod["machines"],
                            dependencies=dependencies,
                            production_order_nr=order.production_order_nr,
                            station_settings={
                                "taste": products[idx][0].setting_taste,
                                "bottle_size": products[idx][0].setting_bottle_size,
                            },
                            amount=amount,
                        )
                    )
        return cls(data=data, jobs=sub_jobs, machines=machines)


class ObjectiveFunction(Enum):
    CUSTOM_OBJECTIVE = 0
    MAKESPAN = 1
    MAXIMUM_LATENESS = 2
    TOTAL_SETUP_TIME = 3


if __name__ == "__main__":
    data = parse_data("examples/data_v1_single.xlsx")
    jssp = JobShopProblem.from_data(data)
