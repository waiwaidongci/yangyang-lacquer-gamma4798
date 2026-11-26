from sqlalchemy import Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from src.config.database import Base


class CalibrationRule(Base):
    __tablename__ = 'calibration_rules'

    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[int] = mapped_column(ForeignKey('calibration_versions.version'), index=True)
    meter: Mapped[str] = mapped_column(String(64))
    starts: Mapped[str] = mapped_column(String(10))
    ends: Mapped[str] = mapped_column(String(10))
    lower: Mapped[float] = mapped_column(Float)
    upper: Mapped[float] = mapped_column(Float)
    created_by: Mapped[int] = mapped_column(ForeignKey('accounts.id'))
    created_at: Mapped[str] = mapped_column(String(32))


class CalibrationVersion(Base):
    __tablename__ = 'calibration_versions'

    version: Mapped[int] = mapped_column(primary_key=True)
    created_by: Mapped[int] = mapped_column(ForeignKey('accounts.id'))
    created_at: Mapped[str] = mapped_column(String(32))
