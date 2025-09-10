# BetGPT Web Application

A modern web interface for the BetGPT horse racing prediction system, featuring real-time race data, machine learning predictions, and betting recommendations.

## Features

- **Real-time Race Data**: Live race information from TAB APIs
- **ML Predictions**: PyTorch neural network predictions for race outcomes
- **Betting Recommendations**: Kelly Criterion-based betting suggestions
- **Modern UI**: Responsive web interface with Bootstrap 5
- **API Integration**: RESTful API for data access

## Quick Start

### Development

1. **Install Dependencies**
   ```bash
   cd web_app
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   python run.py
   ```

3. **Access the App**
   Open your browser to `http://localhost:5000`

### Production Deployment

#### Using Docker

1. **Build and Run**
   ```bash
   docker-compose up -d
   ```

2. **Access the App**
   Open your browser to `http://localhost`

#### Manual Deployment

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   ```bash
   export FLASK_ENV=production
   export SECRET_KEY=your-secret-key-here
   ```

3. **Run with WSGI Server**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
   ```

## API Endpoints

### Race Data
- `GET /api/race/<date>/<meet_no>/<race_no>` - Get race details
- `GET /api/predictions/<date>/<meet_no>/<race_no>` - Get model predictions

### Recommendations
- `POST /api/recommendations` - Get betting recommendations

### Example Request
```json
{
  "date": "2025-01-20",
  "meet_no": 2,
  "race_no": 1,
  "bet_type": "place",
  "market": "fixed",
  "model_weight": 0.2,
  "min_edge": 0.0,
  "bankroll": 100.0,
  "kelly_fraction": 0.25
}
```

## Configuration

### Environment Variables
- `FLASK_ENV` - Flask environment (development/production)
- `SECRET_KEY` - Flask secret key for sessions
- `FLASK_DEBUG` - Enable debug mode (development only)

### Model Files
The application expects the following model files in the parent directory:
- `artifacts/model_regression.pth` - PyTorch model
- `artifacts/preprocess.pkl` - Preprocessing artifacts
- `five_year_dataset.parquet` - Training dataset

## Architecture

### Services
- **DataService**: Handles TAB API integration and data processing
- **RecommendationService**: Generates betting recommendations using ML predictions

### Frontend
- **Bootstrap 5**: Responsive UI framework
- **Chart.js**: Data visualization
- **Vanilla JavaScript**: Application logic

### Backend
- **Flask**: Web framework
- **PyTorch**: Machine learning model
- **Pandas**: Data processing
- **Requests**: HTTP client for API calls

## Development

### Project Structure
```
web_app/
├── app.py                 # Main Flask application
├── run.py                 # Development server
├── wsgi.py                # Production WSGI entry point
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
├── templates/             # HTML templates
│   ├── base.html
│   └── index.html
├── static/                # Static assets
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
└── services/              # Business logic
    ├── data_service.py
    └── recommendation_service.py
```

### Adding New Features

1. **API Endpoints**: Add routes in `app.py`
2. **Business Logic**: Create services in `services/`
3. **Frontend**: Update templates and JavaScript
4. **Styling**: Modify CSS in `static/css/`

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure model files exist in parent directory
   - Check file permissions
   - Verify PyTorch installation

2. **API Connection Issues**
   - Check network connectivity
   - Verify TAB API endpoints are accessible
   - Review request headers and authentication

3. **Performance Issues**
   - Monitor memory usage
   - Consider caching for frequently accessed data
   - Optimize database queries if applicable

### Logs
Application logs are written to stdout. In production, configure log aggregation as needed.

## Security Considerations

- Change default secret keys in production
- Use HTTPS in production environments
- Implement rate limiting for API endpoints
- Validate all user inputs
- Keep dependencies updated

## License

This project is for educational and research purposes only. Please ensure compliance with local gambling regulations.

## Disclaimer

This system is for educational and research purposes only. Betting involves risk and should be done responsibly. Past performance does not guarantee future results.
