# caregiver.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from models import Patient, Caregiver, patient_caregiver
from db import get_db  

router = APIRouter()

class InviteRequest(BaseModel):
    patient_email: EmailStr
    caregiver_email: EmailStr
    caregiver_name: str

@router.post("/invite", status_code=200)
def invite_caregiver(req: InviteRequest, db: Session = Depends(get_db)):
    # 1) Obtener o crear paciente
    patient = db.query(Patient).filter_by(email=req.patient_email).first()
    if not patient:
        raise HTTPException(404, "Paciente no encontrado")
    # 2) Obtener o crear cuidador
    caregiver = db.query(Caregiver).filter_by(email=req.caregiver_email).first()
    if not caregiver:
        caregiver = Caregiver(name=req.caregiver_name, email=req.caregiver_email)
        db.add(caregiver)
        db.commit()
        db.refresh(caregiver)
    # 3) Añadir relación (invitado pero aún no aceptado)
    stmt = patient_caregiver.insert().values(
        patient_id=patient.id,
        caregiver_id=caregiver.id
    )
    db.execute(stmt)
    db.commit()
    return {"message": "Invitación enviada", "caregiver_id": caregiver.id}

class AcceptRequest(BaseModel):
    patient_id: int
    caregiver_id: int

@router.post("/accept", status_code=200)
def accept_invite(req: AcceptRequest, db: Session = Depends(get_db)):
    # Actualiza accepted_at en la fila asociativa
    upd = patient_caregiver.update()\
        .where(
            (patient_caregiver.c.patient_id == req.patient_id) &
            (patient_caregiver.c.caregiver_id == req.caregiver_id)
        )\
        .values(accepted_at=func.now())
    result = db.execute(upd)
    if result.rowcount == 0:
        raise HTTPException(404, "Invitación no encontrada")
    db.commit()
    return {"message": "Invitación aceptada"}

@router.get("/patients/{caregiver_id}")
def list_patients(caregiver_id: int, db: Session = Depends(get_db)):
    cg = db.query(Caregiver).filter_by(id=caregiver_id).first()
    if not cg:
        raise HTTPException(404, "Cuidador no encontrado")
    return [{"id": p.id, "name": p.name, "email": p.email} for p in cg.patients]

@router.get("/caregivers/{patient_id}")
def list_caregivers(patient_id: int, db: Session = Depends(get_db)):
    pt = db.query(Patient).filter_by(id=patient_id).first()
    if not pt:
        raise HTTPException(404, "Paciente no encontrado")
    return [{"id": c.id, "name": c.name, "email": c.email} for c in pt.caregivers]

