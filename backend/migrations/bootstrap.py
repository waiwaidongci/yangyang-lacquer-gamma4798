from sqlalchemy import text

from src.config.database import Base, engine
from src.models import account, calibration, dataset, review, run  # noqa: F401


def bootstrap():
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        columns = {row[1] for row in connection.execute(text('PRAGMA table_info(analysis_runs)'))}
        if 'calibration_version' not in columns:
            connection.execute(text('ALTER TABLE analysis_runs ADD COLUMN calibration_version INTEGER NOT NULL DEFAULT 0'))
