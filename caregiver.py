from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
import random

from models import Patient, Caregiver, patient_caregiver, Invitation
from db import get_db   # Importa aquí, no desde main.py

router = APIRouter(prefix="/caregiver", tags=["caregiver"])

class InviteRequest(BaseModel):
    patient_email: EmailStr
    caregiver_email: EmailStr
    caregiver_name: str

class CodeRequest(BaseModel):
    patient_email: EmailStr

class JoinRequest(BaseModel):
    code: str
    caregiver_email: EmailStr
    caregiver_name: str

@router.post("/invite", status_code=200)
def invite_caregiver(req: InviteRequest, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter_by(email=req.patient_email).first()
    if not patient:
        raise HTTPException(404, "Paciente no encontrado")

    caregiver = db.query(Caregiver).filter_by(email=req.caregiver_email).first()
    if not caregiver:
        caregiver = Caregiver(name=req.caregiver_name, email=req.caregiver_email)
        db.add(caregiver)
        db.commit()
        db.refresh(caregiver)

    stmt = patient_caregiver.insert().values(
        patient_id=patient.id,
        caregiver_id=caregiver.id
    )
    db.execute(stmt)
    db.commit()
    return {"message": "Invitación enviada", "caregiver_id": caregiver.id}

@router.post("/code", status_code=200)
def generate_code(req: CodeRequest, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter_by(email=req.patient_email).first()
    if not patient:
        raise HTTPException(404, "Paciente no encontrado")

    # Genera un código único de 6 dígitos
    code = None
    while True:
        cand = f"{random.randint(0, 999999):06d}"
        if not db.query(Invitation).filter_by(code=cand).first():
            code = cand
            break

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

    db.delete(inv)
    db.commit()
    return {"message": "Unido correctamente como cuidador", "patient_id": patient.id}
