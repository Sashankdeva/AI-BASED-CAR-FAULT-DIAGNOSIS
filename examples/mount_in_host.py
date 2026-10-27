"""Option B: mount this module inside a shared FastAPI host app (one process, one port).

    python examples/mount_in_host.py
    -> http://localhost:8000/car-fault/ui/  and  http://localhost:8000/car-fault/api/v1/health
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from fastapi import FastAPI

from api.main import create_app

host = FastAPI(title="Team Platform")
host.mount("/car-fault", create_app())
# host.mount("/hazard-detection", other_team_app) ...


@host.get("/")
def home():
    return {"modules": {"car-fault-diagnosis": "/car-fault/api/v1/manifest"}}


if __name__ == "__main__":
    uvicorn.run(host, port=8000)
