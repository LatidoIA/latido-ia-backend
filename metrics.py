from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import get_db
from models import Metric

router = APIRouter(tags=["metrics"])

class MetricIn(BaseModel):
    patient_id: int
    metric: str
    value: float
    unit: str
    timestamp: Optional[datetime] = None
    source: str

class MetricOut(BaseModel):
    metric: str
    value: float
    unit: str
    timestamp: datetime
    source: str

@router.post("/metrics", status_code=201, response_model=MetricOut)
def create_metric(payload: MetricIn, db: Session = Depends(get_db)):
    m = Metric(**payload.dict())
    db.add(m)
    db.commit()
    db.refresh(m)
    return m

@router.get(
    "/caregiver/patient/{patient_id}/metrics",
    response_model=List[MetricOut]
)
def get_metrics(
    patient_id: int,
    from_ts: Optional[datetime] = Query(None, alias="from"),
    to_ts:   Optional[datetime] = Query(None, alias="to"),
    db:      Session        = Depends(get_db)
):
    query = db.query(Metric).filter(Metric.patient_id == patient_id)
    if from_ts:
        query = query.filter(Metric.timestamp >= from_ts)
    if to_ts:
        query = query.filter(Metric.timestamp <= to_ts)
    metrics = query.order_by(Metric.timestamp.desc()).all()
    if not metrics:
        # opcional: devolver [] en lugar de 404
        return []
    return metrics
