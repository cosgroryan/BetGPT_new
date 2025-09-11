# Docker Compose Commands for BetGPT Web App

## Modern Docker Compose v2 Commands

### Basic Commands
```bash
# Build and start all services
docker compose up -d

# Build services
docker compose build

# Start services (without building)
docker compose start

# Stop services
docker compose stop

# Stop and remove containers
docker compose down

# Stop, remove containers, and remove volumes
docker compose down -v
```

### Development Commands
```bash
# Build and start with logs
docker compose up --build

# View logs
docker compose logs -f

# View logs for specific service
docker compose logs -f betgpt-web
docker compose logs -f nginx

# Execute command in running container
docker compose exec betgpt-web bash
docker compose exec nginx sh
```

### Service Management
```bash
# Check service status
docker compose ps

# Restart specific service
docker compose restart betgpt-web
docker compose restart nginx

# Scale services (if needed)
docker compose up -d --scale betgpt-web=2
```

### Debugging Commands
```bash
# Check service health
docker compose ps

# View detailed logs
docker compose logs --tail=100 betgpt-web

# Check nginx configuration
docker compose exec nginx nginx -t

# Test connectivity between services
docker compose exec nginx ping betgpt-web
```

### Production Commands
```bash
# Build for production
docker compose -f docker-compose.yml build --no-cache

# Start in production mode
docker compose -f docker-compose.yml up -d

# Update services
docker compose pull
docker compose up -d
```

### Cleanup Commands
```bash
# Remove unused images
docker image prune

# Remove unused volumes
docker volume prune

# Remove unused networks
docker network prune

# Remove everything unused
docker system prune -a
```

## Service Names
- `betgpt-web`: Flask application
- `nginx`: Nginx reverse proxy

## Ports
- `80`: Nginx (external access)
- `443`: Nginx HTTPS (external access)
- `8080`: Flask app (internal only)

## Health Checks
- Application: `http://localhost/health`
- Direct Flask: `http://localhost:8080/health` (internal)

## Troubleshooting

### Service won't start
```bash
# Check logs
docker compose logs betgpt-web

# Check if ports are in use
netstat -tulpn | grep :80
netstat -tulpn | grep :8080
```

### Nginx issues
```bash
# Test nginx config
docker compose exec nginx nginx -t

# Reload nginx config
docker compose exec nginx nginx -s reload
```

### Flask app issues
```bash
# Check Flask logs
docker compose logs betgpt-web

# Access Flask container
docker compose exec betgpt-web bash

# Check if Flask is responding
docker compose exec betgpt-web curl localhost:8080/health
```
