import math

from fastapi import HTTPException
from sqlalchemy import select

from src.constants.workflow import Decision, Role, RunStatus
from src.domain.handover import ESTIMATE_TABLES, HANDOVER_TABLES, handover_table
from src.domain.project import SPEC
from src.models.review import Review
from src.models.run import Run
from src.services import calibration
from src.services.catalog import get_dataset
from src.services.execution import calculate
from src.utils.time import utc_stamp


def run_info(run, review, current_version=None):
    fields = ('id', 'label', 'dataset_id', 'created_at', 'status', 'settings')
    return {**{key: getattr(run, key) for key in fields}, 'decision': review.decision if review else None,
            'calibration_version': run.calibration_version,
            'calibration_stale': current_version is not None and run.calibration_version != current_version}


def list_runs(session, actor):
    version, _ = calibration.current(session)
    statement = select(Run, Review).outerjoin(Review, Run.id == Review.run_id).order_by(Run.id.desc()).limit(100)
    if actor.role == Role.VIEWER:
        statement = statement.where(Review.decision == Decision.APPROVED)
    return [run_info(run, review, version) for run, review in session.execute(statement)]


def get_run(session, run_id, actor):
    run = session.get(Run, run_id)
    review = session.get(Review, run_id)
    if run is None or (actor.role == Role.VIEWER and (not review or review.decision != Decision.APPROVED)):
        raise LookupError('找不到可查看的运行')
    return run, review


def validate_settings(settings):
    parameters = {item['key']: item for item in SPEC['settings']}
    if settings.keys() - parameters.keys():
        raise ValueError('存在本项目未定义的运行参数')
    checked = {}
    for key, parameter in parameters.items():
        value = settings.get(key, parameter['default'])
        if not math.isfinite(value) or not parameter['min'] <= value <= parameter['max']:
            raise ValueError(f"{parameter['label']}超出范围")
        if parameter['step'] == 1 and value != int(value):
            raise ValueError(f"{parameter['label']}必须是整数")
        checked[key] = int(value) if parameter['step'] == 1 else value
    return checked


def create_run(session, payload, actor):
    rows = get_dataset(session, payload.dataset_id).rows
    actor_id = actor.id
    settings = validate_settings(payload.settings)
    created_at = utc_stamp()
    version, rules = calibration.current(session)
    session.rollback()
    result, error = calculate(rows, settings, rules)
    run = Run(label=payload.label, dataset_id=payload.dataset_id, created_by=actor_id,
              created_at=created_at, settings=settings, result=result, error=error,
              status=RunStatus.FAILED if error is not None else RunStatus.COMPLETED,
              calibration_version=version)
    session.add(run)
    session.commit()
    return run_info(run, None, version)


def export_view(session, run_id, key, actor):
    run, review = get_run(session, run_id, actor)
    if review is None or review.decision != Decision.APPROVED:
        raise HTTPException(409, '复核通过后可下载正式 CSV')
    if not any(table['key'] == key for table in run.result['tables']):
        raise LookupError('找不到结果表')
    status = calibration.handover(session, run)
    if status['training_stale'] and key in ESTIMATE_TABLES:
        raise HTTPException(409, '训练用校准规则已更新，旧模型估时已失效，请按当前校准重跑后再交接。')
    paused = status['paused_orders']
    if paused and key in HANDOVER_TABLES:
        table = handover_table(run.result['tables'], key, set(paused))
    else:
        table = next(table for table in run.result['tables'] if table['key'] == key)
    return table, paused
