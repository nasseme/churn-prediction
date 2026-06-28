# app.py
import subprocess
import sys
import os

# Lance l'API en arrière-plan
api = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.main:app",
                        "--host", "0.0.0.0", "--port", "8000"])

# Lance Streamlit au premier plan
os.environ["API_URL"] = "http://localhost:8000"
subprocess.run([sys.executable, "-m", "streamlit", "run",
                "src/app/streamlit_app.py",
                "--server.port=7860",
                "--server.address=0.0.0.0"])