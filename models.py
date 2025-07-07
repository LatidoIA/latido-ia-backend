# models.py
from sqlalchemy import Column, Integer, String, Table, ForeignKey, DateTime, Float, func
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

patient_caregiver = Table(
    "patient_caregiver", Base.metadata,
    Column("patient_id", ForeignKey("patients.id"), primary_key=True),
    Column("caregiver_id", ForeignKey("caregivers.id"), primary_key=True),
    Column("invited_at", DateTime, server_default=func.now()),
    Column("accepted_at", DateTime, nullable=True),
)

class Patient(Base):
    __tablename__ = "patients"
    id    = Column(Integer, primary_key=True, index=True)
    name  = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    caregivers = relationship(
      "Caregiver", secondary=patient_caregiver, back_populates="patients"
    )

class Caregiver(Base):
    __tablename__ = "caregivers"
    id    = Column(Integer, primary_key=True, index=True)
    name  = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    patients = relationship(
      "Patient", secondary=patient_caregiver, back_populates="caregivers"
    )

class Invitation(Base):
    __tablename__ = "invitations"
    id         = Column(Integer, primary_key=True, index=True)
    code       = Column(String(6), unique=True, nullable=False)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    created_at = Column(DateTime, server_default=func.now())

class Metric(Base):
    __tablename__ = "metrics"
    id         = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    metric     = Column(String, nullable=False)       # p.ej. "heart_rate"
    value      = Column(Float, nullable=False)
    unit       = Column(String, nullable=False)       # p.ej. "bpm"
    timestamp  = Column(DateTime, server_default=func.now())
    source     = Column(String, nullable=False)       # p.ej. "healthkit"
