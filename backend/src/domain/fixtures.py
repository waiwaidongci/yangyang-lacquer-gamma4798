"""Synthetic work orders for the repair room's November handover."""

from datetime import date, timedelta
from math import exp
from random import Random

from .records import Row, normalize_rows


def demo_rows() -> list[Row]:
    rng = Random(18344)
    first = date(2026, 11, 1)
    rows = []
    for day in range(1, 21):
        work_day = first + timedelta(days=day - 1)
        for lot in (1, 2):
            # A jar prepared on the 15th was used on both sides of the default cutoff.
            batch = "QP-1115-X" if (day, lot) in ((15, 2), (16, 1)) else f"QP-11{day:02d}-{lot}"
            batch_offset = rng.gauss(0, 0.07) if batch != "QP-1115-X" else 0.06
            for part in range(1, 4):
                film = round(rng.uniform(28, 105), 1)
                temperature = round(rng.uniform(17, 27), 1)
                humidity = round(rng.uniform(37, 75), 1)
                mean = 8 * exp(0.013 * (film - 55) - 0.060 * (temperature - 22)
                               + 0.026 * (humidity - 55) + batch_offset)
                hours = round(rng.gammavariate(16, mean / 16), 2)
                rows.append({
                    "id": f"KP-11{day:02d}-{lot}{part}",
                    "work_date": work_day.isoformat(),
                    "prep_batch": batch,
                    "record_type": "history",
                    "film_um": film,
                    "temperature_c": temperature,
                    "humidity_pct": humidity,
                    "planned_hours": 10 if film < 75 else 14,
                    "dry_hours": hours,
                })

    schedules = (
        (25, "QP-1125-A", 74, 20, 67, 12),
        (25, "QP-1125-A", 52, 23, 54, 12),
        (25, "QP-1125-B", 92, 19, 73, 14),
        (25, "QP-1125-B", 64, 21, 61, 10),
        (26, "QP-1126-A", 48, 24, 46, 10),
        (26, "QP-1126-A", 59, 22, 58, 10),
        (26, "QP-1126-B", 111, 18, 77, 14),
        (26, "QP-1126-B", 79, 22, 64, 14),
    )
    for number, (day, batch, film, temperature, humidity, reserved) in enumerate(schedules, 1):
        rows.append({
            "id": f"KP-11{day:02d}-P{number:02d}",
            "work_date": date(2026, 11, day).isoformat(),
            "prep_batch": batch,
            "record_type": "planned",
            "film_um": film,
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "planned_hours": reserved,
            "dry_hours": None,
        })
    return normalize_rows(rows)
