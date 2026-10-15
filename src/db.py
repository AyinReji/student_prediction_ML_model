from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

DB_PATH = Path("data/student_support.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

Base = declarative_base()
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

PROGRESS_CAVEAT = (
    "Note on interpretation: Before/after changes do not prove that a follow-up caused the change. "
    "Observed improvements or declines can occur due to test variability, regression to the mean, "
    "or external factors not captured in this data."
)


class Student(Base):
    __tablename__ = "students"

    student_id = Column(Integer, primary_key=True)
    age = Column(Integer, nullable=False)
    gender = Column(String(20), nullable=False)
    study_hours_per_day = Column(Float, nullable=False)
    attendance_percentage = Column(Float, nullable=False)
    social_media_hours = Column(Float, nullable=False)
    series_1 = Column(Float, nullable=False)
    series_2 = Column(Float, nullable=False)
    final_exam_score = Column(Float, nullable=True)

    followups = relationship("FollowUp", back_populates="student", cascade="all, delete-orphan")
    assessments = relationship("Assessment", back_populates="student", cascade="all, delete-orphan")


class FollowUp(Base):
    __tablename__ = "followups"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.student_id"), nullable=False)
    date = Column(DateTime, default=datetime.utcnow, nullable=False)
    teacher = Column(String(100), nullable=False)
    action_type = Column(String(100), nullable=False)
    notes = Column(Text, nullable=True)
    status = Column(String(50), default="Completed", nullable=False)

    student = relationship("Student", back_populates="followups")


class Assessment(Base):
    __tablename__ = "assessments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.student_id"), nullable=False)
    assessment_name = Column(String(100), nullable=False)
    date = Column(DateTime, default=datetime.utcnow, nullable=False)
    score = Column(Float, nullable=False)
    max_score = Column(Float, default=40.0, nullable=False)

    student = relationship("Student", back_populates="assessments")


def init_db(force_reload: bool = False) -> None:
    """Initializes SQLite tables and seeds baseline student records and assessments."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)

    session = SessionLocal()
    try:
        count = session.query(Student).count()
        if count == 0 or force_reload:
            raw_path = Path("data/raw/student_performance_10000-1.csv")
            if not raw_path.exists():
                raw_path = Path("student_performance_10000-1.csv")
            df = pd.read_csv(raw_path)

            students_to_add = []
            assessments_to_add = []
            now = datetime.utcnow()

            for idx, row in df.iterrows():
                sid = int(idx)
                s = Student(
                    student_id=sid,
                    age=int(row["age"]),
                    gender=str(row["gender"]),
                    study_hours_per_day=float(row["study_hours_per_day"]),
                    attendance_percentage=float(row["attendance_percentage"]),
                    social_media_hours=float(row["social_media_hours"]),
                    series_1=float(row["series_1"]),
                    series_2=float(row["series_2"]),
                    final_exam_score=float(row["final_exam_score"]),
                )
                students_to_add.append(s)

                # Add initial series 1 & 2 as baseline assessments
                a1 = Assessment(
                    student_id=sid,
                    assessment_name="Series 1 Exam",
                    date=now,
                    score=float(row["series_1"]),
                    max_score=40.0,
                )
                a2 = Assessment(
                    student_id=sid,
                    assessment_name="Series 2 Exam",
                    date=now,
                    score=float(row["series_2"]),
                    max_score=40.0,
                )
                assessments_to_add.append(a1)
                assessments_to_add.append(a2)

            session.bulk_save_objects(students_to_add)
            session.bulk_save_objects(assessments_to_add)
            session.commit()
    finally:
        session.close()


def log_followup(
    student_id: int,
    teacher: str,
    action_type: str,
    notes: str,
    status: str = "Completed",
    follow_date: datetime | None = None,
) -> int:
    """Logs a follow-up action for a student."""
    session = SessionLocal()
    try:
        followup = FollowUp(
            student_id=student_id,
            teacher=teacher,
            action_type=action_type,
            notes=notes,
            status=status,
            date=follow_date or datetime.utcnow(),
        )
        session.add(followup)
        session.commit()
        session.refresh(followup)
        return followup.id
    finally:
        session.close()


def get_followups(student_id: int | None = None) -> list[dict[str, Any]]:
    """Retrieves follow-ups optionally filtered by student_id."""
    session = SessionLocal()
    try:
        query = session.query(FollowUp)
        if student_id is not None:
            query = query.filter(FollowUp.student_id == student_id)
        results = query.order_by(FollowUp.date.desc()).all()
        return [
            {
                "id": r.id,
                "student_id": r.student_id,
                "date": r.date.strftime("%Y-%m-%d %H:%M"),
                "teacher": r.teacher,
                "action_type": r.action_type,
                "notes": r.notes,
                "status": r.status,
            }
            for r in results
        ]
    finally:
        session.close()


def log_assessment(
    student_id: int,
    assessment_name: str,
    score: float,
    max_score: float = 40.0,
    assess_date: datetime | None = None,
) -> int:
    """Logs a subsequent assessment score for progress tracking."""
    session = SessionLocal()
    try:
        assessment = Assessment(
            student_id=student_id,
            assessment_name=assessment_name,
            score=score,
            max_score=max_score,
            date=assess_date or datetime.utcnow(),
        )
        session.add(assessment)
        session.commit()
        session.refresh(assessment)
        return assessment.id
    finally:
        session.close()


def get_student_assessments(student_id: int) -> list[dict[str, Any]]:
    """Retrieves all assessment milestones for a student."""
    session = SessionLocal()
    try:
        assessments = (
            session.query(Assessment)
            .filter(Assessment.student_id == student_id)
            .order_by(Assessment.date.asc(), Assessment.id.asc())
            .all()
        )
        return [
            {
                "id": a.id,
                "assessment_name": a.assessment_name,
                "date": a.date.strftime("%Y-%m-%d %H:%M"),
                "score": a.score,
                "max_score": a.max_score,
                "pct": round((a.score / a.max_score) * 100, 1),
            }
            for a in assessments
        ]
    finally:
        session.close()


def get_student_progress(student_id: int) -> dict[str, Any]:
    """Evaluates student trajectory before and after the earliest logged follow-up."""
    session = SessionLocal()
    try:
        first_followup = (
            session.query(FollowUp)
            .filter(FollowUp.student_id == student_id)
            .order_by(FollowUp.date.asc())
            .first()
        )

        all_assessments = (
            session.query(Assessment)
            .filter(Assessment.student_id == student_id)
            .order_by(Assessment.date.asc(), Assessment.id.asc())
            .all()
        )

        if not all_assessments:
            return {
                "student_id": student_id,
                "has_followup": False,
                "assessments": [],
                "before_avg_pct": None,
                "after_avg_pct": None,
                "delta_pct": None,
                "caveat": PROGRESS_CAVEAT,
            }

        assess_list = [
            {
                "name": a.assessment_name,
                "score": a.score,
                "max_score": a.max_score,
                "pct": (a.score / a.max_score) * 100,
                "date": a.date,
            }
            for a in all_assessments
        ]

        if first_followup is None:
            # Baseline trajectory (Series 1 -> Series 2)
            s1 = assess_list[0]["pct"] if len(assess_list) > 0 else 0
            s2 = assess_list[1]["pct"] if len(assess_list) > 1 else s1
            return {
                "student_id": student_id,
                "has_followup": False,
                "assessments": assess_list,
                "before_avg_pct": round(s1, 1),
                "after_avg_pct": round(s2, 1),
                "delta_pct": round(s2 - s1, 1),
                "caveat": PROGRESS_CAVEAT,
            }

        cutoff_date = first_followup.date
        before = [a["pct"] for a in assess_list if a["date"] <= cutoff_date]
        after = [a["pct"] for a in assess_list if a["date"] > cutoff_date]

        before_avg = np.mean(before) if before else np.mean([a["pct"] for a in assess_list[:2]])
        after_avg = np.mean(after) if after else None
        delta = round(after_avg - before_avg, 2) if after_avg is not None else None

        return {
            "student_id": student_id,
            "has_followup": True,
            "first_followup_date": cutoff_date.strftime("%Y-%m-%d"),
            "assessments": assess_list,
            "before_avg_pct": round(float(before_avg), 1) if before_avg is not None else None,
            "after_avg_pct": round(float(after_avg), 1) if after_avg is not None else None,
            "delta_pct": delta,
            "caveat": PROGRESS_CAVEAT,
        }
    finally:
        session.close()
