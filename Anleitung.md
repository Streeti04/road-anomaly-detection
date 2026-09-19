# Anleitung (PowerShell)

- Git und Python 3.12 installieren, Projekt klonen und öffnen:
  ```powershell
  git clone https://github.com/Streeti04/road-anomaly-detection.git
  cd road-anomaly-detection
  ```
- Python-Umgebung, Pakete und Modellgewichte einrichten:
  ```powershell
  py -3.12 -m venv .venv
  $py = ".\.venv\Scripts\python.exe"
  & $py -m pip install -r requirements.txt
  & $py -m aibip.fetch_models
  ```
- Eigene Pfade zu Bildern und zugehörigen Positionsmetadaten setzen:
  ```powershell
  $bilder = "D:\Cyclomedia\Bilder"
  $metadaten = "D:\Cyclomedia\recordings_bbox_complete.json"
  ```
- Tests und Bestandsprüfung ausführen; bei Fehlern zuerst diese beheben:
  ```powershell
  & $py -m pytest -q tests/test_portable_pool.py
  & $py -m aibip.pool inventory --images $bilder --metadata $metadaten
  ```
- Masken erzeugen, danach Overlays unter `outputs/qc/test_c` prüfen:
  ```powershell
  & $py -m aibip.pool segment --images $bilder --metadata $metadaten
  ```
- Patches erzeugen, danach `outputs/pool/patches_accepted.jpg` prüfen:
  ```powershell
  & $py -m aibip.pool extract --images $bilder --metadata $metadaten
  ```
- Modell trainieren (`auto`: CUDA, falls verfügbar, sonst CPU):
  ```powershell
  & $py -m aibip.pool train --epochs 30 --device auto
  ```
- Rangliste erzeugen; Ergebnisse unter `outputs/pool/runs/mse/stage2/pool_analysis` ansehen:
  ```powershell
  & $py -m aibip.pool analyze --device auto
  ```

---

# Anleitung (zsh / macOS)

- Git und Python 3.12 installieren (per Homebrew), Projekt klonen und öffnen:
  ```zsh
  brew install git python@3.12
  git clone https://github.com/Streeti04/road-anomaly-detection.git
  cd road-anomaly-detection
  ```
- Python-Umgebung, Pakete und Modellgewichte einrichten:
  ```zsh
  python3.12 -m venv .venv
  py=".venv/bin/python"
  "$py" -m pip install -r requirements.txt
  "$py" -m aibip.fetch_models
  ```
- Eigene Pfade zu Bildern und zugehörigen Positionsmetadaten setzen (hier: Symlink `data` auf die externe Platte):
  ```zsh
  bilder="data/26-07-14 Cyclomedia-Test-Hammersbach"
  metadaten="data/26-07-14 Cyclomedia-Test-Hammersbach/recordings_bbox_complete.json"
  ```
- Tests und Bestandsprüfung ausführen; bei Fehlern zuerst diese beheben:
  ```zsh
  "$py" -m pytest -q tests/test_portable_pool.py
  "$py" -m aibip.pool inventory --images "$bilder" --metadata "$metadaten"
  ```
- Masken erzeugen, danach Overlays unter `outputs/qc/test_c` prüfen:
  ```zsh
  "$py" -m aibip.pool segment --images "$bilder" --metadata "$metadaten"
  ```
- Patches erzeugen, danach `outputs/pool/patches_accepted.jpg` prüfen:
  ```zsh
  "$py" -m aibip.pool extract --images "$bilder" --metadata "$metadaten"
  ```
- Modell trainieren (`auto`: CUDA, falls verfügbar, sonst MPS (Apple GPU), sonst CPU):
  ```zsh
  "$py" -m aibip.pool train --epochs 30 --device auto
  ```
- Rangliste erzeugen; Ergebnisse unter `outputs/pool/runs/mse/stage2/pool_analysis` ansehen:
  ```zsh
  "$py" -m aibip.pool analyze --device auto
  ```
