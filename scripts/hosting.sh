#!/usr/bin/env bash
# Public hosting of the MedXVLM demo from this PC (its GPU), through an ngrok tunnel.
#
#   scripts/hosting.sh install   # one-time: create + enable the two systemd --user services
#   scripts/hosting.sh start     # start app + tunnel (they also start automatically when WSL starts)
#   scripts/hosting.sh stop      # stop both (e.g. before gaming, to give the GPU back)
#   scripts/hosting.sh restart | status | url | logs
#
# Two services, both restarted automatically if they crash:
#   medxvlm-app     Streamlit on 127.0.0.1:8501 (loopback only; the tunnel is the only way in)
#   medxvlm-tunnel  ngrok http -> public HTTPS URL (the account's free static domain, if it has one)
# The app frees the ~7 GB report model after MEDXVLM_IDLE_UNLOAD_SECONDS of inactivity; the 4-model
# classifier ensemble (~1.4 GB) stays loaded while the app runs.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"
PY_ENV="$HOME/miniconda3/envs/ml"
NGROK="$(command -v ngrok || echo /usr/local/bin/ngrok)"
PORT=8501
IDLE_SECONDS=600

write_units() {
  mkdir -p "$UNIT_DIR"
  cat > "$UNIT_DIR/medxvlm-app.service" <<EOF
[Unit]
Description=MedXVLM Streamlit demo (public via medxvlm-tunnel)
After=network-online.target

[Service]
Type=simple
Environment=PYTHONPATH=src
Environment=MEDXVLM_IDLE_UNLOAD_SECONDS=${IDLE_SECONDS}
Environment=PYTHONUNBUFFERED=1
ExecStart=/bin/bash -c "cd '${REPO}' && exec '${PY_ENV}/bin/streamlit' run app/app.py --server.port ${PORT} --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false"
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF
  cat > "$UNIT_DIR/medxvlm-tunnel.service" <<EOF
[Unit]
Description=ngrok tunnel for the MedXVLM demo
After=network-online.target medxvlm-app.service
Wants=medxvlm-app.service

[Service]
Type=simple
ExecStart=${NGROK} http 127.0.0.1:${PORT} --log stdout --log-format logfmt
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
}

public_url() {
  curl -s --max-time 3 http://127.0.0.1:4040/api/tunnels 2>/dev/null \
    | python3 -c 'import json,sys; t=json.load(sys.stdin).get("tunnels",[]); print(next((x["public_url"] for x in t if x["public_url"].startswith("https")), ""))' 2>/dev/null || true
}

case "${1:-status}" in
  install)
    write_units
    loginctl enable-linger "$USER" >/dev/null 2>&1 || echo "note: could not enable lingering (services stop when you log out of WSL)"
    systemctl --user enable medxvlm-app.service medxvlm-tunnel.service
    echo "installed; start with: $0 start" ;;
  start)
    systemctl --user start medxvlm-app.service medxvlm-tunnel.service
    for _ in $(seq 1 30); do u=$(public_url); [ -n "$u" ] && break; sleep 1; done
    echo "public URL: ${u:-<not up yet; try: $0 url>}" ;;
  stop)
    systemctl --user stop medxvlm-tunnel.service medxvlm-app.service
    echo "stopped (GPU memory released)" ;;
  restart)
    systemctl --user restart medxvlm-app.service medxvlm-tunnel.service
    sleep 5; echo "public URL: $(public_url)" ;;
  url)
    public_url ;;
  logs)
    journalctl --user -u medxvlm-app.service -u medxvlm-tunnel.service -n "${2:-40}" --no-pager ;;
  status)
    for s in medxvlm-app medxvlm-tunnel; do printf '%-16s %s\n' "$s" "$(systemctl --user is-active $s.service 2>/dev/null)"; done
    printf '%-16s %s\n' "public URL" "$(public_url)"
    command -v nvidia-smi >/dev/null && printf '%-16s %s\n' "GPU memory" "$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)" ;;
  *)
    echo "usage: $0 install|start|stop|restart|status|url|logs [n]"; exit 1 ;;
esac
