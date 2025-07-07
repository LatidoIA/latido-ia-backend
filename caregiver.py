import random
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

from models import Patient, Caregiver, patient_caregiver, Invitation
from db import get_db

router = APIRouter(prefix="/caregiver", tags=["caregiver"])


class CodeRequest(BaseModel):
    patient_email: EmailStr


class JoinRequest(BaseModel):
    code: str
    caregiver_email: EmailStr
    caregiver_name: str


class PatientResponse(BaseModel):
    id: int
    name: str
    email: EmailStr


@router.post("/code", status_code=200)
def generate_code(req: CodeRequest, db: Session = Depends(get_db)):
    # 1) Si no existe el paciente, créalo
    patient = db.query(Patient).filter_by(email=req.patient_email).first()
    if not patient:
        patient = Patient(
            name=req.patient_email.split('@')[0],
            email=req.patient_email
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)

    # 2) Generar código único de 6 dígitos
    code = None
    while True:
        candidate = f"{random.randint(0, 999999):06d}"
        if not db.query(Invitation).filter_by(code=candidate).first():
            code = candidate
            break

    # 3) Guardar invitación
    inv = Invitation(code=code, patient_id=patient.id)
    db.add(inv)
    db.commit()

    return {"code": code}


@router.post("/join", status_code=200)
def join_with_code(req: JoinRequest, db: Session = Depends(get_db)):
    # 1) Buscar invitación
    inv = db.query(Invitation).filter_by(code=req.code).first()
    if not inv:
        raise HTTPException(status_code=404, detail="Código inválido")

    # 2) Obtener paciente asociado
    patient = db.query(Patient).get(inv.patient_id)

    # 3) Crear caregiver si no existe
    caregiver = db.query(Caregiver).filter_by(email=req.caregiver_email).first()
    if not caregiver:
        caregiver = Caregiver(
            name=req.caregiver_name,
            email=req.caregiver_email
        )
        db.add(caregiver)
        db.commit()
        db.refresh(caregiver)

    # 4) Asociar paciente ↔ caregiver
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

    # 5) Eliminar la invitación
    db.delete(inv)
    db.commit()

    return {
        "message": "Unido correctamente como cuidador",
        "patient_id": patient.id
        "patient_name": patient.name  
    }


@router.get("/patient", response_model=PatientResponse)
def get_patient(email: EmailStr, db: Session = Depends(get_db)):
    """
    Recupera el paciente por su email.
    """
    patient = db.query(Patient).filter_by(email=email).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")
    return patient
