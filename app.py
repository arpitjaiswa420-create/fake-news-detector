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

# Check if being executed via Streamlit Cloud or `streamlit run app.py`
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is not None:
        ui_script_path = PROJECT_ROOT / "ui" / "app.py"
        with open(ui_script_path, "r", encoding="utf-8") as f:
            code = compile(f.read(), str(ui_script_path), "exec")
            exec(code, globals())
        sys.exit(0)
except Exception:
    pass

from api.app import app

if __name__ == "__main__":
    import uvicorn
    print("Starting Fake News Detector REST API on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
