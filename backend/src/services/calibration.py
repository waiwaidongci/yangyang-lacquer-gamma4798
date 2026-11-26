from sqlalchemy import func, select

from src.domain.calibration import affected_orders, normalize_rules
from src.models.calibration import CalibrationRule, CalibrationVersion
from src.services.catalog import get_dataset
from src.utils.time import utc_stamp

RULE_KEYS = ('meter', 'starts', 'ends', 'lower', 'upper')


def _as_dict(rule):
    return {key: getattr(rule, key) for key in RULE_KEYS}


def _rules_at(session, version):
    statement = (select(CalibrationRule).where(CalibrationRule.version == version)
                 .order_by(CalibrationRule.starts, CalibrationRule.ends, CalibrationRule.meter, CalibrationRule.id))
    return [_as_dict(rule) for rule in session.scalars(statement)]


def current(session):
    """当前校准版本及其规则；从未登记时版本为 0、规则为空。"""
    version = session.scalar(select(func.max(CalibrationVersion.version))) or 0
    return version, _rules_at(session, version) if version else []


def view(session):
    version, rules = current(session)
    return {'version': version, 'rules': rules}


def replace(session, payload, actor):
    """整组替换校准规则并递增版本；与当前规则完全相同则不产生新版本。"""
    rules = normalize_rules([item.model_dump() for item in payload.rules])
    version, existing = current(session)
    if rules == existing:
        return {'version': version, 'rules': existing}
    version += 1
    created_at = utc_stamp()
    session.add(CalibrationVersion(version=version, created_by=actor.id, created_at=created_at))
    for rule in rules:
        session.add(CalibrationRule(version=version, created_by=actor.id, created_at=created_at, **rule))
    session.commit()
    return {'version': version, 'rules': rules}


def handover(session, run):
    """运行的交接状态：训练读数是否失效、哪些待排班工单须暂停估时。"""
    version, rules = current(session)
    status = {'calibration_version': run.calibration_version, 'current_version': version,
              'training_stale': False, 'paused_orders': []}
    if run.calibration_version != version:
        rows = get_dataset(session, run.dataset_id).rows
        old_rules = _rules_at(session, run.calibration_version) if run.calibration_version else []
        training, paused = affected_orders(rows, old_rules, rules)
        status['training_stale'] = training
        status['paused_orders'] = paused
    return status
