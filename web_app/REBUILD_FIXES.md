# 🔧 Rebuild Fixes Summary

This document summarizes all the fixes included in the `rebuild.sh` script.

## 🎯 Issues Fixed

### 1. **Model Path Issues** ✅
- **Problem**: Model files not found - `[Errno 2] No such file or directory: 'artifacts/preprocess.pkl'`
- **Root Cause**: Model prediction functions using relative paths instead of absolute paths
- **Fix**: Updated all model prediction calls to use absolute paths:
  - `model_path="/app/artifacts/model_regression.pth"`
  - `artefacts_path="/app/artifacts/preprocess.pkl"`

### 2. **Python Import Path Issues** ✅
- **Problem**: `ModuleNotFoundError: No module named 'recommend_picks_NN'`
- **Root Cause**: Python path not correctly set up for Docker environment
- **Fix**: Updated `app.py` to detect Docker vs local environment and set paths accordingly:
  - Docker: `/app/scripts/web_app_scripts` and `/app/scripts/helper_scripts`
  - Local: `web_app/scripts/web_app_scripts` and `web_app/scripts/helper_scripts`

### 3. **Port Configuration** ✅
- **Problem**: Flask app running on wrong port, health checks failing
- **Fix**: 
  - Flask app runs on port 8080
  - Nginx runs on port 80
  - Health checks use correct port 8080

### 4. **Static File Serving** ✅
- **Problem**: Nginx 404 errors for static files (CSS, JS, favicon)
- **Root Cause**: Nginx trying to serve static files directly instead of proxying to Flask
- **Fix**: Updated nginx.conf to proxy static file requests to Flask app

### 5. **Docker Compose v2** ✅
- **Problem**: Using deprecated `docker-compose` command
- **Fix**: Updated all scripts to use `docker compose` (v2 syntax)

### 6. **Nginx Configuration** ✅
- **Problem**: Nginx container restarting due to configuration issues
- **Fix**: 
  - Complete nginx.conf file with proper upstream configuration
  - Correct proxy settings for all routes
  - Proper static file handling

## 📁 Files Modified

### Core Application Files
- `app.py` - Updated model prediction calls with absolute paths
- `recommend_picks_NN.py` - Added model path parameters
- `pytorch_pre.py` - Functions now accept absolute paths

### Configuration Files
- `docker-compose.yml` - Port mappings, volume mounts, health checks
- `nginx.conf` - Complete reverse proxy configuration
- `wsgi.py` - Port 8080 configuration

### Scripts
- `deploy.sh` - Docker Compose v2, correct health check URLs
- `rebuild.sh` - Comprehensive rebuild with all fixes
- `deploy-nginx.sh` - Systemd deployment option

## 🚀 Deployment Commands

### Quick Rebuild (Recommended)
```bash
cd ~/BetGPT_new/web_app
./rebuild.sh
```

### Full Deployment
```bash
cd ~/BetGPT_new/web_app
./deploy.sh
```

### Manual Commands
```bash
# Stop containers
docker compose down

# Rebuild and start
docker compose up -d --build

# Check status
docker compose ps

# View logs
docker compose logs -f betgpt-web
```

## 🧪 Verification Tests

After rebuild, verify:
1. **Health Check**: `curl http://localhost/health`
2. **Static Files**: `curl http://localhost/static/css/style.css`
3. **Model Predictions**: Load a race and check for model percentages
4. **Results Feature**: Test the results button functionality

## 📊 Expected Results

- ✅ Model percentages showing in runners table
- ✅ Static files loading (CSS, JS, favicon)
- ✅ Results button working with win/place odds
- ✅ No more "No such file or directory" errors
- ✅ Nginx serving as proper reverse proxy
- ✅ All containers healthy and stable

## 🔍 Troubleshooting

If issues persist:
1. Check container logs: `docker compose logs`
2. Verify model files: `docker compose exec betgpt-web ls -la /app/artifacts/`
3. Test model prediction: Use the debugging commands in the main README
4. Check Nginx config: `docker compose exec nginx nginx -t`
