#!/usr/bin/env bash
# Stant başlatıcı: oyunu kiosk modunda (tam ekran, fare gizli) çalıştırır.
# Oyun çökerse 3 sn sonra yeniden başlatır; operatör ESC'yi basılı tutarak
# çıkarsa (çıkış kodu 0) döngü biter.
#
# Ortam değişkenleriyle değiştirilebilir:
#   OPPONENTS=flybrain,heuristic   (virgülle birden fazla: ziyaretçi seçer)
#   LANG_UI=tr                     (arayüz dili; oyunda F3 ile de değişir)
#   CONFIGS="configs/brain_tuned.toml configs/stand.toml"
# Örnek: LANG_UI=tr ./kiosk.sh
set -u
cd "$(dirname "$0")"
OPPONENTS="${OPPONENTS:-flybrain,heuristic}"
LANG_UI="${LANG_UI:-en}"
CONFIGS="${CONFIGS:-configs/brain_tuned.toml configs/stand.toml}"
args=()
for c in $CONFIGS; do args+=(--config "$c"); done
while true; do
  .venv/bin/python -m flygame play --kiosk --opponent "$OPPONENTS" --lang "$LANG_UI" "${args[@]}" "$@"
  code=$?
  if [ "$code" -eq 0 ]; then
    break
  fi
  echo "Game exited with code $code; restarting in 3 s (Ctrl+C to stop)" >&2
  sleep 3
done
