"""Persist and reload run artefacts with joblib (reproducibility)."""
import os

import joblib

from config import ARTIFACT_DIR


def save_run(run: dict, name="latest_run.joblib", folder=ARTIFACT_DIR):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    joblib.dump(run, path)
    return path


def load_run(name="latest_run.joblib", folder=ARTIFACT_DIR):
    return joblib.load(os.path.join(folder, name))
