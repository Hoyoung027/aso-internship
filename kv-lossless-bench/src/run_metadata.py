"""공통 결과 식별자: 한국 시간_실험 종류_job 번호."""
import json
import os
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def new_metadata():
    # Slurm에서는 실제 job 번호를 사용하고, 직접 실행은 local-PID로 구분한다.
    timestamp = os.environ.get("KV_RUN_TIMESTAMP") or datetime.now(ZoneInfo("Asia/Seoul")).strftime("%y%m%d_%H%M%S")
    datetime.strptime(timestamp, "%y%m%d_%H%M%S")
    if not re.fullmatch(r"\d{6}_\d{6}", timestamp):
        raise ValueError("KV_RUN_TIMESTAMP must be YYMMDD_HHMMSS")
    job_id = os.environ.get("SLURM_JOB_ID") or f"local-{os.getpid()}"
    experiment = os.environ.get("KV_EXPERIMENT", "exp-a")
    for value in (job_id, experiment):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("Unsafe job ID or experiment name")
    return dict(run_id=f"{timestamp}_{experiment}_job-{job_id}",
                timestamp=timestamp, timezone="Asia/Seoul", job_id=job_id,
                experiment=experiment)


def source_metadata(source):
    # 오프라인 재분석도 KV를 생성한 실행의 식별자를 유지한다.
    path = Path(source) / "run.json"
    if not path.exists():
        raise ValueError("Missing source run.json; result provenance is required")
    run = json.loads(path.read_text())
    return {key: run[key] for key in ("run_id", "timestamp", "timezone", "job_id", "experiment")}


if __name__ == "__main__":
    print(new_metadata()["run_id"])
