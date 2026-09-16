#!/usr/bin/env zsh
# Usage: ./train.sh                     komplette Pipeline (Tests → Training → Analyse)
#        SKIP_SEGMENT=1 ./train.sh      Masken-Schritt überspringen (schon gelaufen)
#        FORCE_SEGMENT=1 ./train.sh     inkompatible alte Masken sichern und neu generieren
#        RESUME=1 ./train.sh            Training ab letztem Checkpoint fortsetzen
#        EPOCHS=10 ./train.sh           andere Epochenzahl (Default: 30)
set -euo pipefail
cd "$(dirname "$0")"

py=".venv/bin/python"
bilder="data/26-07-14 Cyclomedia-Test-Hammersbach"
metadaten="data/26-07-14 Cyclomedia-Test-Hammersbach/recordings_bbox_complete.json"

mkdir -p outputs/logs
log="outputs/logs/train_$(date +%Y%m%d_%H%M%S).log"

step() {
  print "\n=== $1 ==="
  print "Start: $(date '+%H:%M:%S')"
}

step "Log"
print "Alles wird zusätzlich geloggt nach: $log"

t0=$SECONDS

step "1/5 Tests"
"$py" -m pytest -q tests/test_portable_pool.py 2>&1 | tee "$log"

step "2/5 Inventory"
"$py" -m aibip.pool inventory --images "$bilder" --metadata "$metadaten" 2>&1 | tee -a "$log"

if [[ ${SKIP_SEGMENT:-0} == 1 ]]; then
  step "3/5 Segment — übersprungen (SKIP_SEGMENT=1)"
elif [[ -d outputs/qc/test_c && -n "$(ls outputs/qc/test_c 2>/dev/null)" ]]; then
  step "3/5 Segment — übersprungen (outputs/qc/test_c nicht leer; Ordner löschen zum Neuerzeugen)"
else
  step "3/5 Segment (längster Schritt; Fortschritt: 'Preflight:'- und 'C Test:'-Zeilen)"
  "$py" -m aibip.pool segment --images "$bilder" --metadata "$metadaten" ${FORCE_SEGMENT:+--force} 2>&1 | tee -a "$log"
fi

if [[ -f outputs/pool/patches_accepted.jpg ]]; then
  step "4/5 Patches — übersprungen (outputs/pool/patches_accepted.jpg existiert)"
else
  step "4/5 Patches erzeugen, danach outputs/pool/patches_accepted.jpg prüfen"
  "$py" -m aibip.pool extract --images "$bilder" --metadata "$metadaten" 2>&1 | tee -a "$log"
fi

step "5/5 Training (${EPOCHS:-30} Epochen, --device auto)"
"$py" -m aibip.pool train --epochs "${EPOCHS:-30}" --device auto ${RESUME:+--resume} 2>&1 | tee -a "$log"

step "Bonus: Rangliste/Analyse"
"$py" -m aibip.pool analyze --device auto 2>&1 | tee -a "$log"

print "\n=== Fertig nach $(( (SECONDS - t0) / 60 )) min ==="
print "Ergebnisse: outputs/pool/runs/mse/stage2/pool_analysis"
print "Log:        $log"
