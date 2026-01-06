# app/models.py
from datetime import date
from pydantic import BaseModel
from typing import Optional, List


class AddOn(BaseModel):
    name: str
    monthly_charge: float


class ContractExtraction(BaseModel):
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None
    customer_address: Optional[str] = None

    plan_name: Optional[str] = None
    plan_charge: Optional[float] = None
    minimum_monthly_plan: Optional[float] = None

    contract_start_date: Optional[date] = None
    contract_end_date: Optional[date] = None

    order_number: Optional[str] = None
    activity: Optional[str] = None

    down_payment: Optional[float] = None

    device_model: Optional[str] = None
    device_imei: Optional[str] = None
    serial_number: Optional[str] = None
    sim_number: Optional[str] = None

    add_ons: List[AddOn] = []


class ExtractResponse(BaseModel):
    success: bool
    message: str
    extraction: Optional[ContractExtraction] = None
