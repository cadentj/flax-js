import os
from pathlib import Path

from dotenv import load_dotenv

REPO_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = os.path.join(REPO_DIR, "benchmark", "output")

load_dotenv(REPO_DIR / ".env")