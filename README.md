# BetGPT - Horse Racing Prediction System

## 📁 Project Structure

This project has been organized into a clean, logical structure for better maintainability and deployment.

### 🏗️ Directory Organization

```
BetGPT_new/
├── web_app/                    # Main web application
│   ├── app.py                 # Flask application
│   ├── run.py                 # Development server runner
│   ├── wsgi.py                # WSGI entry point
│   ├── requirements.txt       # Web app dependencies
│   ├── Dockerfile            # Container configuration
│   ├── docker-compose.yml    # Multi-container setup
│   ├── deploy.sh             # Deployment script
│   ├── env.example           # Environment variables template
│   ├── README.md             # Web app documentation
│   ├── DEPLOYMENT.md         # Deployment guide
│   │
│   ├── services/              # Business logic services
│   │   ├── data_service.py    # TAB API integration
│   │   └── recommendation_service.py # Betting recommendations
│   │
│   ├── static/                # Static web assets
│   │   ├── css/
│   │   │   └── style.css      # Custom styling
│   │   └── js/
│   │       └── app.js         # Frontend JavaScript
│   │
│   ├── templates/             # HTML templates
│   │   ├── base.html          # Base template
│   │   └── index.html         # Main page
│   │
│   └── scripts/               # Web app specific scripts
│       ├── web_app_scripts/   # Core ML and prediction scripts
│       │   ├── recommend_picks_NN.py
│       │   ├── pytorch_pre.py
│       │   ├── model_features.py
│       │   ├── update_data_retrain_model.py
│       │   └── start_web_app.sh
│       │
│       ├── helper_scripts/    # Utility and helper scripts
│       │   ├── NEW_racing_GUI.py
│       │   ├── lgbm_rank.py
│       │   ├── race_data.py
│       │   ├── schedule_scraper.py
│       │   ├── schedule_flattener.py
│       │   ├── fetch_tab_results.py
│       │   ├── flatten_race_json_nd.py
│       │   ├── infer_from_schedule_json.py
│       │   ├── merge_parquet.py
│       │   ├── pull_gallops_range.py
│       │   ├── check_file.py
│       │   ├── check_parquet.py
│       │   ├── csv_join.py
│       │   ├── head.py
│       │   └── import argparse.py
│       │
│       └── old_scripts/       # Legacy and deprecated scripts
│           ├── racing_gui.py
│           ├── benchmark_yesterday.py
│           ├── backtest_blend.py
│           ├── eval_holdout.py
│           ├── evaluate_one_race.py
│           ├── recs_by_meet.py
│           ├── run_backtest_sweep.py
│           ├── dayslip_gallops_only.py
│           ├── table_top_wool_solver_no_limits copy.py
│           └── table_top_wool_solver_with_limits.py
│
├── data/                      # All data files organized by type
│   ├── csvs/                 # CSV files by category
│   │   ├── benchmarks/       # Benchmark results
│   │   │   ├── 2 model benchmark_gallops_2025-08-12.csv
│   │   │   ├── benchmark_gallops_2025-08-12.csv
│   │   │   ├── benchmark_gallops_2025-08-13.csv
│   │   │   ├── benchmark_gallops_2025-08-16.csv
│   │   │   └── benchmark_gallops_2025-08-17.csv
│   │   │
│   │   ├── backtest_results/ # Backtest outputs
│   │   │   ├── backtest_grid_results_2week.csv
│   │   │   └── backtest_grid_results.csv
│   │   │
│   │   ├── sweep_results/    # Parameter sweep results
│   │   │   ├── sweep_results_20250819_072144.csv
│   │   │   ├── sweep_results_20250819_074043.csv
│   │   │   └── sweep_results_20250819_075458.csv
│   │   │
│   │   ├── race_data/        # Individual race data
│   │   │   ├── M2_Ruakaka_R2_2025-08-16_13-08-18.csv
│   │   │   └── M2_Ruakaka_R6_2025-08-16_13-08-56.csv
│   │   │
│   │   └── other/           # Miscellaneous CSVs
│   │       ├── eval_log.csv
│   │       ├── first_100_rows.csv
│   │       ├── first_1000_rows.csv
│   │       ├── wool_allocation_contract_summary.csv
│   │       └── wool_allocation_tolerance.csv
│   │
│   ├── parquet/             # Parquet data files
│   │   ├── five_year_dataset.parquet
│   │   ├── old_five_year_dataset.parquet
│   │   ├── gallops_2024-07-30_to_2024-07-31.parquet
│   │   ├── gallops_2025-06-01_to_2025-06-30.parquet
│   │   ├── gallops_2025-07-01_to_2025-07-15.parquet
│   │   ├── gallops_2025-08-03_to_2025-08-10.parquet
│   │   ├── gallops_2025-08-17_to_2025-08-17.parquet
│   │   ├── gallops_2025-08-18_to_2025-08-18.parquet
│   │   └── gallops_2025-08-19_to_2025-08-19.parquet
│   │
│   ├── json/                # JSON configuration and data
│   │   ├── race.json
│   │   ├── schedule_data_example.json
│   │   └── metrics.json
│   │
│   ├── results/             # Race results data
│   │   └── 2025-08-17/      # Date-specific results
│   │       ├── results_flat.csv
│   │       └── raw/         # Raw JSON results (500+ files)
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
│   ├── picks_betplace_w0p2_blendlogit_orrproportional_me0p6_mk1p0_2week.csv
│   ├── picks_betplace_w0p2_blendlogit_orrproportional_me0p6_mk1p0.csv
│   └── picks_betwin_w0p4_blendlogit_orrnone_me0p04_mk0p0.csv
│
├── backups/               # Data backups
│   └── five_year_dataset_20250815_124142.parquet
│
├── build/                 # Build artifacts (PyInstaller)
│   └── RacingGUI/
│       ├── warn-RacingGUI.txt
│       └── xref-RacingGUI.html
│
├── dist/                  # Distribution files
│   ├── RacingGUI
│   └── RacingGUI.app/
│
├── dayslips/             # Dayslip data (39 files)
│   ├── 2025-08-19_dayslip.txt
│   ├── BATHURST meet29 1-7 2025-08-18 11-27-03.txt
│   ├── CASINO meet22 1-8 2025-08-18 11-32-39.txt
│   ├── dayslip_2025-08-19_*.txt
│   ├── dayslip_combined_2025-08-*.txt
│   ├── dayslip_lgbm_rank_2025-08-*.txt
│   ├── dayslip_lgbm_reg_2025-08-*.txt
│   ├── dayslip_nn_2025-08-*.txt
│   ├── dayslip_starred_2025-08-*.txt
│   └── meet*_*.txt
│
├── dumps/                # API data dumps
│   ├── odds_2025-08-19.json
│   └── schedule_2025-08-19.json
│
├── example_json/         # Example JSON files
│   ├── odds.json
│   ├── results.json
│   └── schedule.json
│
├── new_app/              # Legacy web app
│   ├── result_example.json
│   └── web_app.py
│
└── README.md             # This file
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

### Web App Scripts (`web_app/scripts/web_app_scripts/`)
Core scripts that power the web application:
- `recommend_picks_NN.py` - Neural network predictions
- `pytorch_pre.py` - PyTorch model training and inference
- `model_features.py` - Feature engineering
- `update_data_retrain_model.py` - Model retraining pipeline
- `start_web_app.sh` - Web app startup script

### Helper Scripts (`web_app/scripts/helper_scripts/`)
Utility scripts for data processing and API interactions:
- `NEW_racing_GUI.py` - TAB API integration functions
- `lgbm_rank.py` - LightGBM model training
- `schedule_scraper.py` - Race schedule data collection
- `fetch_tab_results.py` - Results data fetching
- `race_data.py` - Race data processing
- `schedule_flattener.py` - Schedule data processing
- `flatten_race_json_nd.py` - JSON data processing
- `infer_from_schedule_json.py` - Schedule inference
- `merge_parquet.py` - Parquet file operations
- `pull_gallops_range.py` - Data range extraction
- `check_file.py` - File validation utilities
- `check_parquet.py` - Parquet file validation
- `csv_join.py` - CSV file operations
- `head.py` - Data preview utilities
- `import argparse.py` - Argument parsing utilities

### Old Scripts (`web_app/scripts/old_scripts/`)
Legacy scripts and deprecated functionality:
- `racing_gui.py` - Original Tkinter GUI
- `benchmark_yesterday.py` - Benchmarking tools
- `backtest_blend.py` - Backtesting functionality
- `eval_holdout.py` - Model evaluation
- `evaluate_one_race.py` - Single race evaluation
- `recs_by_meet.py` - Recommendations by meet
- `run_backtest_sweep.py` - Parameter sweep backtesting
- `dayslip_gallops_only.py` - Dayslip processing
- `table_top_wool_solver*.py` - Wool allocation solvers

## 📈 Data Organization

### CSV Files (`data/csvs/`)
- **Benchmarks** (`benchmarks/`): Model performance comparisons
  - `benchmark_gallops_2025-08-*.csv` - Daily benchmark results
  - `2 model benchmark_gallops_2025-08-12.csv` - Multi-model comparisons
- **Backtest Results** (`backtest_results/`): Historical strategy performance
  - `backtest_grid_results.csv` - Grid search results
  - `backtest_grid_results_2week.csv` - Two-week backtest results
- **Sweep Results** (`sweep_results/`): Parameter optimization outputs
  - `sweep_results_20250819_*.csv` - Parameter sweep results
- **Race Data** (`race_data/`): Individual race information
  - `M2_Ruakaka_R*_2025-08-16_*.csv` - Specific race data
- **Other** (`other/`): Miscellaneous data files
  - `eval_log.csv` - Evaluation logs
  - `first_100_rows.csv`, `first_1000_rows.csv` - Data samples
  - `wool_allocation_*.csv` - Wool allocation data

### Parquet Files (`data/parquet/`)
- **Main Dataset**: `five_year_dataset.parquet` - Primary training data
- **Backup**: `old_five_year_dataset.parquet` - Previous version
- **Date Ranges**: `gallops_YYYY-MM-DD_to_YYYY-MM-DD.parquet` - Time-specific data
  - Covers periods from 2024-07-30 to 2025-08-19

### JSON Files (`data/json/`)
- **Configuration**: `race.json`, `schedule_data_example.json` - API endpoints, model parameters
- **Metrics**: `metrics.json` - Model performance measurements

### Results Data (`data/results/`)
- **Date-specific Results**: `2025-08-17/` - Organized by date
  - `results_flat.csv` - Flattened results data
  - `raw/` - Raw JSON results (500+ individual race files)

### Visualizations (`data/visualizations/`)
- **HTML Reports**: `3model_vis.html`, `model_effeciency_vis.html`, `visuals.html`

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
python3 web_app/scripts/web_app_scripts/update_data_retrain_model.py --retrain
```

## 📝 Notes

- All paths have been updated to work with the new structure
- The web app automatically finds scripts in the organized directories
- Legacy scripts are preserved but moved to `old_scripts/`
- Data files are organized by type for easy management
- Model artifacts remain in their original locations for compatibility
