# caregiver.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# DTOs sencillos
class CaregiverLink(BaseModel):
    patient_id: str
    caregiver_email: str

# Endpoints de ejemplo

@router.post("/share")
async def share_metrics(link: CaregiverLink):
    """
    El paciente solicita compartir sus métricas con un cuidador.
    Deberás aquí guardar en tu base de datos la relación.
    """
    # TODO: validar patient_id existe y guardar el enlace
    return {"message": f"Métricas del paciente {link.patient_id} compartidas con {link.caregiver_email}"}

@router.get("/list/{patient_id}")
async def list_caregivers(patient_id: str):
    """
    Obtener lista de cuidadores autorizados para un paciente.
    """
    # TODO: recuperar de BD
    return {"patient_id": patient_id, "caregivers": []}

@router.delete("/revoke")
async def revoke_caregiver(link: CaregiverLink):
    """
    Revocar acceso de un cuidador.
    """
    # TODO: eliminar de BD
    return {"message": f"Acceso revocado para {link.caregiver_email} sobre {link.patient_id}"}

