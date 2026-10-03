"""
Root entry point for Fake News Detector API application.
Allows running with:
    python app.py
or:
    uvicorn app:app --port 8000
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.app import app

if __name__ == "__main__":
    import uvicorn
    print("Starting Fake News Detector REST API on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
