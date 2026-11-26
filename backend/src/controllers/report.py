from fastapi.responses import Response

from src.constants.workflow import RunStatus
from src.services import calibration
from src.services.reviews import review_info
from src.services.runs import export_view, get_run, run_info
from src.utils.csv_files import write_rows


def report_detail(session, run_id, actor):
    run, review = get_run(session, run_id, actor)
    version, _ = calibration.current(session)
    handover = calibration.handover(session, run) if run.status == RunStatus.COMPLETED else None
    return {**run_info(run, review, version), 'result': run.result, 'error': run.error,
            'review': review_info(review) if review else None, 'handover': handover}


def report_csv(session, run_id, key, actor):
    table, paused = export_view(session, run_id, key, actor)
    headers = {'Content-Disposition': f'attachment; filename="run-{run_id}-{table["key"]}.csv"'}
    if paused:
        headers['X-Paused-Orders'] = ','.join(paused)
    return Response(write_rows(table['columns'], table['rows']), media_type='text/csv', headers=headers)
