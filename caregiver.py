from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from typing import Optional
from sqlalchemy.orm import Session

from models import Patient, Caregiver, Invitation, patient_caregiver
from db import get_db
import random

router = APIRouter(prefix="/caregiver", tags=["caregiver"])

# Petición para generar código, ahora acepta nombre opcional
class CodeRequest(BaseModel):
    patient_email: EmailStr
    patient_name: Optional[str] = None

class JoinRequest(BaseModel):
    code: str
    caregiver_email: EmailStr
    caregiver_name: str

@router.post("/code", status_code=200)
def generate_code(req: CodeRequest, db: Session = Depends(get_db)):
    # Busca o crea paciente, usando nombre si viene
    patient = db.query(Patient).filter_by(email=req.patient_email).first()
    if not patient:
        name_to_use = req.patient_name or req.patient_email.split('@')[0]
        patient = Patient(name=name_to_use, email=req.patient_email)
        db.add(patient)
        db.commit()
        db.refresh(patient)
    else:
        # Actualiza nombre si ha cambiado
        if req.patient_name and patient.name != req.patient_name:
            patient.name = req.patient_name
            db.commit()
            db.refresh(patient)

    # Genera código único de 6 dígitos
    code = None
    while True:
        candidate = f"{random.randint(0, 999999):06d}"
        if not db.query(Invitation).filter_by(code=candidate).first():
            code = candidate
            break

    # Guarda invitación
    inv = Invitation(code=code, patient_id=patient.id)
    db.add(inv)
    db.commit()
    return {"code": code}

@router.post("/join", status_code=200)
def join_with_code(req: JoinRequest, db: Session = Depends(get_db)):
    inv = db.query(Invitation).filter_by(code=req.code).first()
    if not inv:
        raise HTTPException(404, "Código inválido")

    patient = db.query(Patient).get(inv.patient_id)

    caregiver = db.query(Caregiver).filter_by(email=req.caregiver_email).first()
    if not caregiver:
        caregiver = Caregiver(name=req.caregiver_name, email=req.caregiver_email)
        db.add(caregiver)
        db.commit()
        db.refresh(caregiver)

    exists = db.execute(
        patient_caregiver.select().where(
            (patient_caregiver.c.patient_id == patient.id) &
            (patient_caregiver.c.caregiver_id == caregiver.id)
        )
    ).first()
    if not exists:
        stmt = patient_caregiver.insert().values(
            patient_id=patient.id,
            caregiver_id=caregiver.id
        )
        db.execute(stmt)
        db.commit()

    # Elimina invitación tras uso
    db.delete(inv)
    db.commit()

    # Retorna también patient_name
    return {
        "message": "Unido correctamente como cuidador",
        "patient_id": patient.id,
        "patient_name": patient.name
    }

