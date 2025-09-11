#!/bin/bash

# BetGPT Web App - Nginx Deployment Script
# This script sets up nginx with the Flask WSGI application

set -e

echo "🚀 Starting BetGPT Web App deployment with Nginx..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_DIR="/app"
NGINX_SITES_DIR="/etc/nginx/sites-available"
NGINX_ENABLED_DIR="/etc/nginx/sites-enabled"
SERVICE_NAME="betgpt-webapp"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root (use sudo)"
   exit 1
fi

# Update system packages
print_status "Updating system packages..."
apt update && apt upgrade -y

# Install required packages
print_status "Installing required packages..."
apt install -y nginx python3 python3-pip python3-venv git

# Create app directory
print_status "Creating application directory..."
mkdir -p $APP_DIR
cd $APP_DIR

# Copy application files (assuming they're in the current directory)
print_status "Copying application files..."
cp -r . $APP_DIR/

# Create virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set proper permissions
print_status "Setting file permissions..."
chown -R www-data:www-data $APP_DIR
chmod -R 755 $APP_DIR

# Configure nginx
print_status "Configuring Nginx..."

# Copy nginx configuration
cp nginx.conf $NGINX_SITES_DIR/betgpt-webapp

# Remove default nginx site if it exists
if [ -f "$NGINX_ENABLED_DIR/default" ]; then
    rm $NGINX_ENABLED_DIR/default
fi

# Enable the site
ln -sf $NGINX_SITES_DIR/betgpt-webapp $NGINX_ENABLED_DIR/

# Test nginx configuration
print_status "Testing Nginx configuration..."
nginx -t

if [ $? -eq 0 ]; then
    print_status "Nginx configuration is valid"
else
    print_error "Nginx configuration test failed"
    exit 1
fi

# Configure systemd service
print_status "Configuring systemd service..."
cp betgpt-webapp.service /etc/systemd/system/

# Reload systemd and enable service
systemctl daemon-reload
systemctl enable $SERVICE_NAME

# Start services
print_status "Starting services..."
systemctl start $SERVICE_NAME
systemctl restart nginx

# Check service status
print_status "Checking service status..."
if systemctl is-active --quiet $SERVICE_NAME; then
    print_status "✅ BetGPT Web App service is running"
else
    print_error "❌ BetGPT Web App service failed to start"
    systemctl status $SERVICE_NAME
    exit 1
fi

if systemctl is-active --quiet nginx; then
    print_status "✅ Nginx is running"
else
    print_error "❌ Nginx failed to start"
    systemctl status nginx
    exit 1
fi

# Show final status
print_status "🎉 Deployment completed successfully!"
echo ""
echo "📋 Service Status:"
systemctl status $SERVICE_NAME --no-pager -l
echo ""
echo "🌐 Nginx Status:"
systemctl status nginx --no-pager -l
echo ""
echo "🔗 Your application should be available at:"
echo "   http://your-server-ip"
echo ""
echo "📝 Useful commands:"
echo "   sudo systemctl status $SERVICE_NAME    # Check app status"
echo "   sudo systemctl restart $SERVICE_NAME   # Restart app"
echo "   sudo systemctl status nginx            # Check nginx status"
echo "   sudo systemctl restart nginx           # Restart nginx"
echo "   sudo nginx -t                          # Test nginx config"
echo "   sudo journalctl -u $SERVICE_NAME -f    # View app logs"
echo "   sudo journalctl -u nginx -f            # View nginx logs"
echo ""
print_warning "Don't forget to:"
echo "   1. Configure your domain name in nginx.conf"
echo "   2. Set up SSL certificates for HTTPS"
echo "   3. Configure firewall rules"
echo "   4. Set up monitoring and logging"
