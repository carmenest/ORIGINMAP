#!/bin/bash
# ORIGINMAP Daily Observation Script
# Runs full pipeline + anomaly detection every 24 hours
#
# To install in cron (runs daily at 3:00 AM):
#   crontab -e
#   0 3 * * * /home/carmenia/originmap/scripts/daily_observation.sh >> /home/carmenia/originmap/logs/cron.log 2>&1
#
# To run manually:
#   ./scripts/daily_observation.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=================================================="
echo "ORIGINMAP Daily Observation - $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=================================================="

# Activate virtual environment
source .venv/bin/activate

# Run full cycle
python -m originmap.cli full-cycle

echo ""
echo "Cycle complete at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=================================================="
