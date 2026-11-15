"""Shared settings. Everyone imports from here so all four parts stay comparable.

Do not change these values on your own branch. Agree in the group first.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"
RESULTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "saved_models"

RAW_FILE = RAW_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"
CLEAN_FILE = PROCESSED_DIR / "clean.csv"

TARGET = "label"

# The UCI dataset page states: label 1 = legitimate URL, label 0 = phishing URL.
PHISHING_LABEL = 0

# Text / identifier columns. They are not fed to the models directly.
TEXT_COLUMNS = ["FILENAME", "URL", "Domain", "TLD", "Title"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
