#!/bin/bash

# Setup script for cron job
# Run this on your server to set up the nightly retraining

echo "Setting up nightly model retraining cron job..."

# Create log directory
sudo mkdir -p /var/log
sudo touch /var/log/betgpt_retrain.log
sudo touch /var/log/betgpt_cron.log

# Set permissions
sudo chown ubuntu:ubuntu /var/log/betgpt_retrain.log
sudo chown ubuntu:ubuntu /var/log/betgpt_cron.log
sudo chmod 644 /var/log/betgpt_retrain.log
sudo chmod 644 /var/log/betgpt_cron.log

# Make the retraining scripts executable
chmod +x /home/ubuntu/BetGPT_new/web_app/scripts/run_nightly_retrain.sh
chmod +x /home/ubuntu/BetGPT_new/web_app/scripts/test_retrain_docker.sh

# Add cron job (runs every night at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * /home/ubuntu/BetGPT_new/web_app/scripts/run_nightly_retrain.sh") | crontab -

echo "Cron job setup complete!"
echo "The model will be retrained every night at 2 AM"
echo "Logs will be written to:"
echo "  - /var/log/betgpt_retrain.log (detailed training logs)"
echo "  - /var/log/betgpt_cron.log (cron execution logs)"
echo ""
echo "To view current cron jobs: crontab -l"
echo "To remove the cron job: crontab -e (then delete the line)"
