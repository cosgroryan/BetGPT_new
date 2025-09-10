# BetGPT - Horse Racing Prediction System

## 📁 Project Structure

This project has been organized into a clean, logical structure for better maintainability and deployment.

### 🏗️ Directory Organization

```
BetGPT_new/
├── web_app/                    # Main web application
│   ├── app.py                 # Flask application
│   ├── services/              # Business logic services
│   ├── static/                # CSS, JS, images
│   ├── templates/             # HTML templates
│   ├── requirements.txt       # Web app dependencies
│   ├── Dockerfile            # Container configuration
│   ├── docker-compose.yml    # Multi-container setup
│   └── DEPLOYMENT.md         # Deployment guide
│
├── scripts/                   # All Python scripts organized by purpose
│   ├── web_app_scripts/      # Scripts used by web application
│   │   ├── recommend_picks_NN.py
│   │   ├── pytorch_pre.py
│   │   ├── model_features.py
│   │   └── update_data_retrain_model.py
│   │
│   ├── helper_scripts/       # Utility and helper scripts
│   │   ├── NEW_racing_GUI.py
│   │   ├── lgbm_rank.py
│   │   ├── race_data.py
│   │   ├── schedule_scraper.py
│   │   ├── schedule_flattener.py
│   │   ├── fetch_tab_results.py
│   │   ├── flatten_race_json_nd.py
│   │   ├── infer_from_schedule_json.py
│   │   ├── merge_parquet.py
│   │   ├── pull_gallops_range.py
│   │   ├── check_file.py
│   │   ├── check_parquet.py
│   │   ├── csv_join.py
│   │   ├── head.py
│   │   └── import argparse.py
│   │
│   └── old_scripts/          # Legacy and deprecated scripts
│       ├── racing_gui.py
│       ├── benchmark_yesterday.py
│       ├── backtest_blend.py
│       ├── eval_holdout.py
│       ├── evaluate_one_race.py
│       ├── recs_by_meet.py
│       ├── run_backtest_sweep.py
│       ├── dayslip_gallops_only.py
│       ├── table_top_wool_solver*.py
│       ├── *.patch
│       ├── *.spec
│       ├── *.icns
│       ├── *.ico
│       └── requirements.txt
│
├── data/                      # All data files organized by type
│   ├── csvs/                 # CSV files by category
│   │   ├── benchmarks/       # Benchmark results
│   │   ├── backtest_results/ # Backtest outputs
│   │   ├── sweep_results/    # Parameter sweep results
│   │   ├── race_data/        # Individual race data
│   │   └── other/           # Miscellaneous CSVs
│   │
│   ├── parquet/             # Parquet data files
│   │   ├── five_year_dataset.parquet
│   │   └── gallops_*.parquet
│   │
│   ├── json/                # JSON configuration and data
│   │   ├── race.json
│   │   ├── schedule_data_example.json
│   │   └── metrics.json
│   │
│   └── visualizations/      # HTML visualization files
│       ├── 3model_vis.html
│       ├── model_effeciency_vis.html
│       └── visuals.html
│
├── artifacts/               # ML model artifacts
│   ├── metrics.json
│   ├── model_regression.pth
│   ├── preprocess.pkl
│   ├── artifacts_pytorch_model.pt
│   └── artifacts_pytorch.pkl
│
├── artifacts_gbm/          # GBM model artifacts
│   ├── metrics_gbm_reg.json
│   ├── metrics_gbm.json
│   ├── model_lgbm_reg.txt
│   ├── model_lgbm.txt
│   ├── preprocess_gbm.pkl
│   └── test_models_exist.py
│
├── backtest_logs/          # Backtest output logs
├── backups/               # Data backups
├── build/                 # Build artifacts
├── dist/                  # Distribution files
├── dayslips/             # Dayslip data
├── dumps/                # API data dumps
├── example_json/         # Example JSON files
└── new_app/              # Legacy web app
```

## 🚀 Quick Start

### Development
```bash
cd web_app
python3 run.py
```

### Production Deployment
```bash
cd web_app
./deploy.sh
```

## 📊 Key Features

- **Web Interface**: Modern Flask-based web application
- **ML Models**: PyTorch neural networks and LightGBM models
- **Real-time Data**: TAB API integration for live race data
- **Betting Recommendations**: Kelly criterion-based stake sizing
- **Visualizations**: Interactive charts and tables
- **Docker Support**: Containerized deployment

## 🔧 Script Categories

### Web App Scripts (`scripts/web_app_scripts/`)
Core scripts that power the web application:
- `recommend_picks_NN.py` - Neural network predictions
- `pytorch_pre.py` - PyTorch model training and inference
- `model_features.py` - Feature engineering
- `update_data_retrain_model.py` - Model retraining pipeline

### Helper Scripts (`scripts/helper_scripts/`)
Utility scripts for data processing and API interactions:
- `NEW_racing_GUI.py` - TAB API integration functions
- `lgbm_rank.py` - LightGBM model training
- `schedule_scraper.py` - Race schedule data collection
- `fetch_tab_results.py` - Results data fetching

### Old Scripts (`scripts/old_scripts/`)
Legacy scripts and deprecated functionality:
- Original GUI applications
- Old backtesting scripts
- Deprecated analysis tools

## 📈 Data Organization

### CSV Files
- **Benchmarks**: Model performance comparisons
- **Backtest Results**: Historical strategy performance
- **Sweep Results**: Parameter optimization outputs
- **Race Data**: Individual race information
- **Other**: Miscellaneous data files

### Parquet Files
- **Main Dataset**: `five_year_dataset.parquet` - Primary training data
- **Date Ranges**: `gallops_YYYY-MM-DD_to_YYYY-MM-DD.parquet` - Time-specific data

### JSON Files
- **Configuration**: API endpoints, model parameters
- **Examples**: Sample data structures
- **Metrics**: Model performance measurements

## 🐳 Deployment

The project includes comprehensive deployment support:

- **Docker**: Containerized application with nginx reverse proxy
- **SSL**: HTTPS configuration with Let's Encrypt
- **Security**: Security headers and CORS configuration
- **Monitoring**: Health checks and logging
- **Scaling**: Multi-worker gunicorn setup

See `web_app/DEPLOYMENT.md` for detailed deployment instructions.

## 🔄 Model Updates

To retrain models with new data:

```bash
python3 scripts/web_app_scripts/update_data_retrain_model.py --retrain
```

## 📝 Notes

- All paths have been updated to work with the new structure
- The web app automatically finds scripts in the organized directories
- Legacy scripts are preserved but moved to `old_scripts/`
- Data files are organized by type for easy management
- Model artifacts remain in their original locations for compatibility
