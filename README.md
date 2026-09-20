# Image Processing — Straßen-Anomalie-Erkennung

Pipeline für ungelabelte Straßenpanoramen: gemessener ROI-Crop und CLAHE (B) →
MobileSAM-Straßenmaske mit Punkt-Prompts (C) → Conv-Autoencoder auf 128×128-
Straßen-Patches (D) → Ranking per Median-Patch-MSE mit Review-Warteschlange (E).

Kein Schadens-Labeling: das Modell lernt „normale Straße“, auffällige Bilder
landen als Review-Signal in `ranking.csv` + Kontaktbögen.

## Stand

- Stage 2 (1.765 Pool-Bilder): 15.155 Patches, 30 Epochen CPU (~4 h),
  beste Epoche 29, Validierungs-MSE 0,001829 — Zahlen und Figures in
  [docs/results](docs/results/README.md).
- 103 Tests in 7 Dateien, nur synthetische Daten.

## Setup

    python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m aibip.fetch_models

Eigene Bilder: [Anleitung.md](Anleitung.md) (frischer Clone, kein Mini-Set nötig).
Architektur: [docs/ARCHITEKTUR.md](docs/ARCHITEKTUR.md), Bericht: [docs/report.md](docs/report.md).
