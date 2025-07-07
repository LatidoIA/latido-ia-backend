from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from sqlalchemy.orm import Session

from db import get_db
from models import Metric

router = APIRouter(prefix="/metrics", tags=["metrics"])

class MetricRequest(BaseModel):
    """
    Modelo genérico para cualquier métrica de paciente.
    """
    patient_id: int = Field(..., description="ID del paciente al que pertenece la métrica")
    metric: str = Field(..., description="Nombre de la métrica (e.g. 'heart_rate', 'blood_pressure', 'mood', 'steps', 'sleep', 'spo2')")
    value: float = Field(..., description="Valor numérico de la métrica")
    unit: str = Field(..., description="Unidad (e.g. 'bpm', 'mmHg', '%', 'count', 'hours')")
    source: str = Field(..., description="Origen de la métrica (e.g. 'healthkit', 'demo', 'wearableSDK')")

@router.post("/", status_code=201)
def create_metric(req: MetricRequest, db: Session = Depends(get_db)):
    """
    Guarda una nueva métrica en la base de datos.
    """
    m = Metric(
        patient_id=req.patient_id,
        metric=req.metric,
        value=req.value,
        unit=req.unit,
        source=req.source,
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return m

@router.get("/patient/{patient_id}", response_model=List[MetricRequest])
def get_metrics(
    patient_id: int,
    from_timestamp: Optional[datetime] = Query(None, alias="from"),
    to_timestamp: Optional[datetime] = Query(None, alias="to"),
    db: Session = Depends(get_db)
):
    """
    Devuelve métricas del paciente ordenadas de más recientes a más antiguas.
    Parámetros opcionales 'from' y 'to' para filtrar por rango de fechas.
    """
    query = db.query(Metric).filter(Metric.patient_id == patient_id)
    if from_timestamp:
        query = query.filter(Metric.timestamp >= from_timestamp)
    if to_timestamp:
        query = query.filter(Metric.timestamp <= to_timestamp)
    metrics = query.order_by(Metric.timestamp.desc()).all()
    return metrics
