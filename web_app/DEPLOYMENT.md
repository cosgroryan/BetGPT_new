# BetGPT Web App - Deployment Guide

## 🚀 Production Deployment Checklist

### 1. **Environment Configuration**

#### Create Production Environment File
```bash
# Create .env file for production
cat > .env << EOF
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your-super-secret-key-here
DATABASE_URL=your-database-url-if-needed
EOF
```

#### Update app.py for Production
```python
# Add to app.py
import os
from dotenv import load_dotenv

load_dotenv()

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
```

### 2. **Dependencies & Requirements**

#### Update requirements.txt
```bash
# Add production dependencies
echo "gunicorn==21.2.0" >> requirements.txt
echo "python-dotenv==1.0.0" >> requirements.txt
echo "psycopg2-binary==2.9.7" >> requirements.txt  # If using PostgreSQL
```

#### Install Production Dependencies
```bash
pip install -r requirements.txt
```

### 3. **WSGI Server Setup**

#### Update wsgi.py
```python
#!/usr/bin/env python3
import os
import sys

# Add the project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

from app import app

if __name__ == "__main__":
    app.run()
```

#### Test WSGI Server
```bash
# Test with gunicorn
gunicorn --bind 0.0.0.0:8080 wsgi:app

# Or with uWSGI
uwsgi --http :8080 --wsgi-file wsgi.py --callable app
```

### 4. **Docker Deployment (Recommended)**

#### Update Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "--timeout", "120", "wsgi:app"]
```

#### Update docker-compose.yml
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8080:8080"
    environment:
      - FLASK_ENV=production
      - FLASK_DEBUG=False
    volumes:
      - ./artifacts:/app/artifacts:ro  # Mount model artifacts
      - ./five_year_dataset.parquet:/app/five_year_dataset.parquet:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro  # SSL certificates
    depends_on:
      - web
    restart: unless-stopped
```

### 5. **Nginx Configuration**

#### Create nginx.conf
```nginx
events {
    worker_connections 1024;
}

http {
    upstream app {
        server web:8080;
    }

    server {
        listen 80;
        server_name your-domain.com;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        # SSL Configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
        ssl_prefer_server_ciphers off;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

        # Gzip compression
        gzip on;
        gzip_vary on;
        gzip_min_length 1024;
        gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;

        location / {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }

        # Static files caching
        location /static/ {
            proxy_pass http://app;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

### 6. **Security Configuration**

#### Update app.py Security
```python
from flask_talisman import Talisman

# Add security headers
Talisman(app, force_https=True)

# CORS configuration for production
from flask_cors import CORS
CORS(app, origins=['https://your-domain.com'])
```

#### Add to requirements.txt
```
flask-talisman==1.1.0
```

### 7. **Database Setup (if needed)**

#### PostgreSQL Setup
```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
CREATE DATABASE betgpt_prod;
CREATE USER betgpt_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE betgpt_prod TO betgpt_user;
\q
```

### 8. **SSL Certificate Setup**

#### Using Let's Encrypt (Certbot)
```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 9. **Monitoring & Logging**

#### Add Logging Configuration
```python
# Add to app.py
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/betgpt.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('BetGPT startup')
```

#### Create logs directory
```bash
mkdir -p logs
```

### 10. **Deployment Commands**

#### Local Testing
```bash
# Test production build
docker-compose up --build

# Test without Docker
gunicorn --bind 0.0.0.0:8080 --workers 4 wsgi:app
```

#### Production Deployment
```bash
# Build and deploy
docker-compose -f docker-compose.prod.yml up -d --build

# View logs
docker-compose logs -f web

# Update application
docker-compose pull
docker-compose up -d --no-deps web
```

### 11. **Environment Variables for Production**

Create `.env.prod`:
```bash
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your-super-secret-production-key
DATABASE_URL=postgresql://user:pass@localhost/betgpt_prod
REDIS_URL=redis://localhost:6379/0
```

### 12. **Backup Strategy**

#### Model Artifacts Backup
```bash
# Create backup script
cat > backup_models.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf "backups/models_${DATE}.tar.gz" artifacts/ five_year_dataset.parquet
aws s3 cp "backups/models_${DATE}.tar.gz" s3://your-backup-bucket/
EOF

chmod +x backup_models.sh
```

### 13. **Performance Optimization**

#### Add Caching
```python
# Add to app.py
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/api/races/<date_str>')
@cache.cached(timeout=300)  # Cache for 5 minutes
def get_races(date_str):
    # ... existing code
```

### 14. **Health Check Endpoint**

```python
# Add to app.py
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })
```

## 🚀 Quick Deployment Commands

```bash
# 1. Prepare environment
cp .env.example .env
# Edit .env with production values

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test locally
gunicorn --bind 0.0.0.0:8080 wsgi:app

# 4. Deploy with Docker
docker-compose up -d --build

# 5. Check status
docker-compose ps
docker-compose logs -f web
```

## 📋 Pre-Deployment Checklist

- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Database configured (if needed)
- [ ] Model artifacts accessible
- [ ] Logging configured
- [ ] Health checks working
- [ ] Security headers enabled
- [ ] Backup strategy in place
- [ ] Monitoring configured
- [ ] Performance testing completed

## 🔧 Troubleshooting

### Common Issues:
1. **Model files not found**: Ensure artifacts/ directory is mounted
2. **CORS errors**: Check CORS configuration for production domain
3. **SSL issues**: Verify certificate paths and permissions
4. **Memory issues**: Increase worker memory or reduce workers
5. **Timeout errors**: Increase proxy timeouts in nginx

### Useful Commands:
```bash
# Check container logs
docker-compose logs web

# Restart services
docker-compose restart web

# Scale workers
docker-compose up -d --scale web=3

# Monitor resources
docker stats
```
