"""Run the full pipeline: python run_all.py   (about 10 minutes)."""
import subprocess
import sys

for step in ("extract.py", "rq1.py", "rq2.py", "rq3.py", "validation.py make"):
    print("==>", step, flush=True)
    subprocess.run([sys.executable, *step.split()], check=True)
