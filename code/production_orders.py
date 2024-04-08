from pydantic import BaseModel
import pandas as pd
import datetime

class Workstation(BaseModel):
    name: str
    max_units_per_run: int
    minutes_per_run: float
    minutes_changeover_time_taste: int
    minutes_changeover_time_bottle_size: int
    starts_at: datetime.time
    stops_at: datetime.time

class Product(BaseModel):
    product_id: int
    product_name: str
    setting_taste: str
    setting_bottle_size: str | None
    workstation_type: str

class BillOfMaterial(BaseModel):
    parent_id: int
    parent_name: str
    component_id: int
    component_name: str
    component_quantity: float

class ProductionOrder(BaseModel):
    production_order_nr: str
    days_till_delivery: int
    product_id: int
    amount: int
    liters_required: float | None = None
    minutes_bottling_time: float | None = None

def parse_data(path: str):
    workstations = pd.read_excel(path, sheet_name="workstations").replace(pd.NA, None).to_dict('records')
    products = pd.read_excel(path, sheet_name="products").replace(pd.NA, None).to_dict('records')
    bill_of_materials = pd.read_excel(path, sheet_name="bill of materials").replace(pd.NA, None).to_dict('records')
    production_orders = pd.read_excel(path, sheet_name="production orders").replace(pd.NA, None).to_dict('records')

    workstations = [Workstation(**workstation) for workstation in workstations] # type: ignore
    products = [Product(**product) for product in products] # type: ignore
    bill_of_materials = [BillOfMaterial(**bom) for bom in bill_of_materials] # type: ignore
    production_orders = [ProductionOrder(**order) for order in production_orders] # type: ignore
    return workstations, products, bill_of_materials, production_orders

if __name__ == "__main__":
    data = parse_data("examples/data_v1.xlsx")
    print(data[0])

    
