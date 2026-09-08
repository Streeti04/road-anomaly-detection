# Image Processing — Straßenbilder-Pipeline

Arbeitsstand. Aufbau: Vorverarbeitung (B) → Straßenmaske (C) → Autoencoder (D) → Ranking (E).

## Setup

    python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

Tests: `pytest -q` (synthetische Daten, kein Modell-Download).
