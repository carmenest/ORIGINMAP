#!/bin/bash
# Installs the daily observation cron job

SCRIPT_PATH="/home/carmenia/originmap/scripts/daily_observation.sh"
LOG_PATH="/home/carmenia/originmap/logs/cron.log"

# Create logs directory
mkdir -p /home/carmenia/originmap/logs

# Remove existing originmap cron entries
crontab -l 2>/dev/null | grep -v "originmap" > /tmp/crontab_new

# Add new entry (runs at 3:00 AM every day)
echo "0 3 * * * $SCRIPT_PATH >> $LOG_PATH 2>&1" >> /tmp/crontab_new

# Install new crontab
crontab /tmp/crontab_new

echo "Cron job installed. Current crontab:"
crontab -l

echo ""
echo "The observation will run daily at 3:00 AM."
echo "Logs will be written to: $LOG_PATH"
echo ""
echo "To change the schedule, run: crontab -e"
echo "To remove, run: crontab -l | grep -v originmap | crontab -"
