from fastapi import APIRouter

from src.controllers.dependencies import Actor, Db, Writer
from src.schemas.calibration import CalibrationUpdate, CalibrationView
from src.services import calibration

router = APIRouter(prefix='/calibration')


@router.get('', response_model=CalibrationView)
def current(session: Db, actor: Actor):
    return calibration.view(session)


@router.put('', response_model=CalibrationView)
def replace(payload: CalibrationUpdate, session: Db, actor: Writer):
    return calibration.replace(session, payload, actor)
