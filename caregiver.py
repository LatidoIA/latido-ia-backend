from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from models import get_db_session, Patient, Caregiver  # asume tu ORM

router = APIRouter(prefix="/caregiver", tags=["caregiver"])

class ShareRequest(BaseModel):
    patient_id: int
    caregiver_email: str

@router.post("/share-link")
async def share_metrics(req: ShareRequest):
    """
    Genera un token/link para que el cuidador acceda a las métricas del paciente.
    """
    # 1) Verificar paciente existe
    patient = await Patient.get(req.patient_id)
    if not patient:
        raise HTTPException(404, "Paciente no encontrado")
    # 2) Crear registro de cuidador
    share = await Caregiver.create_link(patient_id=req.patient_id, email=req.caregiver_email)
    # 3) Devolver URL única (e.g. con UUID)
    return {"share_url": f"https://orca-app-njfej.ondigitalocean.app/caregiver/{share.token}"}

@router.get("/{token}", response_model=PatientMetrics)
async def get_shared_metrics(token: str):
    """
    Devuelve las métricas del paciente asociadas al token de cuidador.
    """
    share = await Caregiver.get_by_token(token)
    if not share or not share.is_active:
        raise HTTPException(404, "Link inválido o expirado")
    patient = await Patient.get(share.patient_id)
    # Aquí selecciona sólo las métricas que quieres exponer:
    return {
        "bpm": patient.latest_bpm,
        "glucose": patient.latest_glucose,
        "sleep_quality": patient.latest_sleep_quality,
        # …
    }
