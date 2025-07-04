# models.py
from sqlalchemy import Column, Integer, String, Table, ForeignKey, DateTime, func
from sqlalchemy.orm import relationship
from db import Base

patient_caregiver = Table(
    "patient_caregiver",
    Base.metadata,
    Column("patient_id", ForeignKey("patients.id"), primary_key=True),
    Column("caregiver_id", ForeignKey("caregivers.id"), primary_key=True),
    Column("invited_at", DateTime, server_default=func.now()),
    Column("accepted_at", DateTime, nullable=True)
)

class Patient(Base):
    __tablename__ = "patients"
    id    = Column(Integer, primary_key=True, index=True)
    name  = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    caregivers = relationship(
        "Caregiver",
        secondary=patient_caregiver,
        back_populates="patients"
    )

class Caregiver(Base):
    __tablename__ = "caregivers"
    id    = Column(Integer, primary_key=True, index=True)
    name  = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    patients = relationship(
        "Patient",
        secondary=patient_caregiver,
        back_populates="caregivers"
    )

        secondary=patient_caregiver,
        back_populates="caregivers"
    )
