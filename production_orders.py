from pydantic import BaseModel
import json
from enum import Enum

class ProductionOrders(BaseModel):
    production_order_nr: int
    days_till_delivery: float
    product_id: int
    amount: int

class Tastes(str, Enum):
    COLA = "cola"
    FANTA = "fanta"
    APPLE_JUICE = "apple_juice"
    ORANGE_JUICE = "orange_juice"

class BottleSizes(float, Enum):
    SMALL = 1.0
    MEDIUM = 1.5
    LARGE = 2.0

class ProductSettings(BaseModel):
    bottle_size: BottleSizes
    taste: Tastes

class Products(BaseModel):
    product_id: int
    product_name: str
    settings: ProductSettings

class OrderSheet(BaseModel):
    production_orders: list[ProductionOrders]
    products: list[Products]

    def get_product(self: "OrderSheet", product_id: int) -> Products:
        if not hasattr(self, "_products_dict"):
            self._products_dict = {product.product_id: product for product in self.products}
        # We could use .get() here, but we want to raise a custom error if the product is not found
        if product_id in self._products_dict:
            return self._products_dict[product_id]
        raise ValueError(f"Product with id {product_id} not found, available product ids: {[product.product_id for product in self.products]}")



if __name__ == "__main__":
    example_json = """
    {
        "production_orders": [
        {"production_order_nr": 1, "days_till_delivery": 1.5, "product_id": 1, "amount": 1000}
        ],
        "products": [
        {"product_id": 1, "product_name": "Cola 2L", "settings": {"bottle_size": 2, "taste": "cola"}},
        {"product_id": 2, "product_name": "Cola 1.5L", "settings": {"bottle_size": 1.5, "taste": "cola"}}
        ]
    }
    """
    json_data = json.loads(example_json)
    order_sheet = OrderSheet(**json_data)
    print(order_sheet.model_dump())