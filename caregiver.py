# caregiver.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel, EmailStr
import random

from models import Patient, Caregiver, patient_caregiver, Invitation
from main import get_db

router = APIRouter(prefix="/caregiver", tags=["caregiver"])


# ——— Pydantic Schemas ———
class InviteRequest(BaseModel):
    patient_email: EmailStr
    caregiver_email: EmailStr
    caregiver_name: str


class AcceptRequest(BaseModel):
    patient_id: int
    caregiver_id: int


class CodeRequest(BaseModel):
    patient_email: EmailStr


class JoinRequest(BaseModel):
    code: str
    caregiver_email: EmailStr
    caregiver_name: str


# ——— Endpoints ———

@router.post("/invite", status_code=200)
def invite_caregiver(req: InviteRequest, db: Session = Depends(get_db)):
    # 1) Verificar paciente existe
    patient = db.query(Patient).filter_by(email=req.patient_email).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")

    # 2) Obtener o crear cuidador
    caregiver = db.query(Caregiver).filter_by(email=req.caregiver_email).first()
    if not caregiver:
        caregiver = Caregiver(
            name=req.caregiver_name,
            email=req.caregiver_email
        )
        db.add(caregiver)
        db.commit()
        db.refresh(caregiver)

    # 3) Asocia invitación (sin accepted_at)
    stmt = patient_caregiver.insert().values(
        patient_id=patient.id,
        caregiver_id=caregiver.id,
        invited_at=func.now()
    )
    db.execute(stmt)
    db.commit()

    return {
        "message": "Invitación enviada",
        "caregiver_id": caregiver.id
    }


@router.post("/accept", status_code=200)
def accept_invite(req: AcceptRequest, db: Session = Depends(get_db)):
    # Actualiza accepted_at en la fila asociativa
    upd = patient_caregiver.update().where(
        (patient_caregiver.c.patient_id == req.patient_id) &
        (patient_caregiver.c.caregiver_id == req.caregiver_id)
    ).values(accepted_at=func.now())
    result = db.execute(upd)
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Invitación no encontrada")
    db.commit()
    return {"message": "Invitación aceptada"}


@router.get("/patients/{caregiver_id}", status_code=200)
def list_patients(caregiver_id: int, db: Session = Depends(get_db)):
    cg = db.query(Caregiver).filter_by(id=caregiver_id).first()
    if not cg:
        raise HTTPException(status_code=404, detail="Cuidador no encontrado")
    return [
        {"id": p.id, "name": p.name, "email": p.email}
        for p in cg.patients
    ]


@router.get("/caregivers/{patient_id}", status_code=200)
def list_caregivers(patient_id: int, db: Session = Depends(get_db)):
    pt = db.query(Patient).filter_by(id=patient_id).first()
    if not pt:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")
    return [
        {"id": c.id, "name": c.name, "email": c.email}
        for c in pt.caregivers
    ]


@router.post("/code", status_code=200)
def generate_code(req: CodeRequest, db: Session = Depends(get_db)):
    # 1) Verificar paciente existe
    patient = db.query(Patient).filter_by(email=req.patient_email).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")

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
    db.refresh(inv)

    return {"code": inv.code}


@router.post("/join", status_code=200)
def join_with_code(req: JoinRequest, db: Session = Depends(get_db)):
    # 1) Buscar invitación
    inv = db.query(Invitation).filter_by(code=req.code).first()
    if not inv:
        raise HTTPException(status_code=404, detail="Código inválido")

    # 2) Obtener paciente
    patient = db.query(Patient).get(inv.patient_id)

    # 3) Obtener o crear cuidador
    caregiver = db.query(Caregiver).filter_by(email=req.caregiver_email).first()
    if not caregiver:
        caregiver = Caregiver(name=req.caregiver_name, email=req.caregiver_email)
        db.add(caregiver)
        db.commit()
        db.refresh(caregiver)

    # 4) Asociar relación si no existe
    exists = db.execute(
        patient_caregiver.select().where(
            (patient_caregiver.c.patient_id == patient.id) &
            (patient_caregiver.c.caregiver_id == caregiver.id)
        )
    ).first()
    if not exists:
        stmt = patient_caregiver.insert().values(
            patient_id=patient.id,
            caregiver_id=caregiver.id,
            accepted_at=func.now()
        )
        db.execute(stmt)
        db.commit()

    # 5) Eliminar invitación usada
    db.delete(inv)
    db.commit()

    return {
        "message": "Unido correctamente como cuidador",
        "patient_id": patient.id
    }

