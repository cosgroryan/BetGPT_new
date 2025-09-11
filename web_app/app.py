#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Horse Racing Prediction Web App
A modern web interface for the BetGPT horse racing prediction system.
"""

import os
import sys
import json
import logging
from datetime import datetime, date
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import pandas as pd
import numpy as np

# Add paths for imports - simplified and robust
current_dir = os.path.dirname(os.path.abspath(__file__))

# Try multiple possible paths for scripts
possible_script_paths = [
    '/app/scripts/web_app_scripts',  # Docker
    '/app/scripts/helper_scripts',   # Docker
    os.path.join(current_dir, 'scripts', 'web_app_scripts'),  # Local
    os.path.join(current_dir, 'scripts', 'helper_scripts'),   # Local
]

# Add existing paths to sys.path
for path in possible_script_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.append(path)

# Set working directory and model paths
if os.path.exists('/app/artifacts'):
    # Docker environment
    parent_dir = '/app'
    MODEL_PATH = '/app/artifacts/model_regression.pth'
    ARTIFACTS_PATH = '/app/artifacts/preprocess.pkl'
else:
    # Local environment - artifacts are in the parent directory of web_app
    parent_dir = os.path.dirname(current_dir)  # This is BetGPT_new
    MODEL_PATH = os.path.join(parent_dir, 'artifacts', 'model_regression.pth')
    ARTIFACTS_PATH = os.path.join(parent_dir, 'artifacts', 'preprocess.pkl')

os.chdir(parent_dir)

# Debug information
print(f"🔍 Debug Info:")
print(f"   Working directory: {os.getcwd()}")
print(f"   Scripts paths added: {[p for p in possible_script_paths if os.path.exists(p)]}")
print(f"   Model path: {MODEL_PATH}")
print(f"   Artifacts path: {ARTIFACTS_PATH}")
print(f"   Model file exists: {os.path.exists(MODEL_PATH)}")
print(f"   Artifacts file exists: {os.path.exists(ARTIFACTS_PATH)}")

# Import model modules
try:
    from recommend_picks_NN import model_win_table
    from pytorch_pre import load_model_and_predict
    print("✅ Model modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"   Available files in /app/scripts:")
    if os.path.exists('/app/scripts'):
        for root, dirs, files in os.walk('/app/scripts'):
            for file in files:
                print(f"     {os.path.join(root, file)}")
    raise

# Import our services
from services.data_service import DataService
from services.recommendation_service import RecommendationService

# Helper functions (extracted from NEW_racing_GUI.py)
def _norm_name_for_join(s):
    """Normalize horse name for matching"""
    if s is None: 
        return ""
    s = str(s).strip().lower()
    s = "".join(ch for ch in s if ch.isalnum() or ch.isspace())
    return " ".join(s.split())

def _imp_from_fixed(win_fx):
    """Calculate implied probability from fixed odds"""
    try:
        fx = float(win_fx)
        return 1.0 / fx if fx > 1.0 else None
    except Exception:
        return None

def make_session():
    """Create a requests session with retry strategy"""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    
    session = requests.Session()
    
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize services
data_service = DataService()
recommendation_service = RecommendationService()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'betgpt-webapp'}), 200

@app.route('/api/races/<date_str>')
def get_races_for_date(date_str):
    """Get all races for a specific date"""
    try:
        session = make_session()
        # This would need to be implemented to fetch all meetings for a date
        # For now, return a placeholder
        return jsonify({
            'date': date_str,
            'meetings': [],
            'message': 'Race fetching not yet implemented'
        })
    except Exception as e:
        logger.error(f"Error fetching races for {date_str}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/race/<date_str>/<int:meet_no>/<int:race_no>')
def get_race_details(date_str, meet_no, race_no):
    """Get detailed race information including runners and odds"""
    try:
        # Fetch race data using data service
        event = data_service.fetch_race_data(date_str, meet_no, race_no)
        
        # Get model predictions
        try:
            model_df = model_win_table(meet_no, race_no, date_str, 
                                     model_path=MODEL_PATH, 
                                     artefacts_path=ARTIFACTS_PATH)
            logger.info(f"Model predictions loaded successfully for {date_str} M{meet_no} R{race_no}")
        except Exception as e:
            logger.warning(f"Model prediction failed: {e}")
            model_df = pd.DataFrame()
            # Continue without model predictions rather than failing
            
        # Process the data for the frontend
        race_data = process_race_data(event, model_df, date_str, meet_no, race_no)
        
        return jsonify(race_data)
        
    except Exception as e:
        logger.error(f"Error fetching race details: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/<date_str>/<int:meet_no>/<int:race_no>')
def get_predictions(date_str, meet_no, race_no):
    """Get model predictions for a specific race"""
    try:
        # Get model predictions
        model_df = model_win_table(meet_no, race_no, date_str, 
                                 model_path="/app/artifacts/model_regression.pth", 
                                 artefacts_path="/app/artifacts/preprocess.pkl")
        
        if model_df.empty:
            return jsonify({'error': 'No predictions available'}), 404
            
        # Convert to JSON-serializable format
        predictions = []
        for _, row in model_df.iterrows():
            predictions.append({
                'runner_number': int(row.get('runner_number', 0)),
                'runner_name': str(row.get('runner_name', '')),
                'win_probability': float(row.get('p_win', 0.0)),
                'win_percentage': float(row.get('win_%', 0.0)),
                'fair_odds': float(row.get('$fair_win', 0.0)),
                'new_horse': bool(row.get('new_horse', False)),
                'fav_rank': int(row.get('fav_rank', 0)) if pd.notna(row.get('fav_rank')) else None
            })
            
        return jsonify({
            'date': date_str,
            'meet_no': meet_no,
            'race_no': race_no,
            'predictions': predictions
        })
        
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get betting recommendations based on model predictions and market odds"""
    try:
        data = request.get_json()
        
        # Extract parameters
        date_str = data.get('date')
        meet_no = int(data.get('meet_no'))
        race_no = int(data.get('race_no'))
        
        # Fetch race data
        event = data_service.fetch_race_data(date_str, meet_no, race_no)
        
        # Get model predictions
        try:
            model_df = model_win_table(meet_no, race_no, date_str, 
                                     model_path=MODEL_PATH, 
                                     artefacts_path=ARTIFACTS_PATH)
        except Exception as e:
            logger.warning(f"Model prediction failed: {e}")
            model_df = pd.DataFrame()
        
        # Process race data
        race_data = process_race_data(event, model_df)
        
        # Generate recommendations using the recommendation service
        recommendations = recommendation_service.generate_recommendations(race_data, data)
        
        return jsonify(recommendations)
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return jsonify({'error': str(e)}), 500

def process_race_data(event, model_df, date_str=None, meet_no=None, race_no=None):
    """Process race event data for frontend consumption"""
    data = event.get("data", {})
    race = data.get("race", {})
    runners = data.get("runners", [])
    
    # Extract race metadata
    race_info = {
        'venue': race.get("display_meeting_name", ""),
        'description': race.get("description", ""),
        'distance': race.get("distance", ""),
        'track_condition': race.get("track_condition", ""),
        'weather': race.get("weather", ""),
        'positions_paid': race.get("positions_paid", 3),
        'start_time': race.get("start_time", ""),
        'country': race.get("meeting_country", ""),
        'date': date_str,
        'meet_no': meet_no,
        'race_no': race_no
    }
    
    # Process runners
    processed_runners = []
    
    # Create model predictions lookup
    model_lookup = {}
    if not model_df.empty:
        for _, row in model_df.iterrows():
            key = _norm_name_for_join(row.get('runner_name', ''))
            model_lookup[key] = {
                'win_prob': float(row.get('p_win', 0.0)),
                'win_percentage': float(row.get('win_%', 0.0)),
                'fair_odds': float(row.get('$fair_win', 0.0)),
                'new_horse': bool(row.get('new_horse', False))
            }
    
    for runner in runners:
        runner_number = runner.get("runner_number") or runner.get("number")
        runner_name = runner.get("name", "")
        
        # Get odds
        prices = runner.get("prices", {})
        odds_obj = runner.get("odds", {})
        
        win_fixed = odds_obj.get("fixed_win", prices.get("win_fixed"))
        place_fixed = odds_obj.get("fixed_place", prices.get("place_fixed"))
        win_tote = prices.get("win_tote")
        place_tote = prices.get("place_tote")
        
        # Get model predictions
        model_key = _norm_name_for_join(runner_name)
        model_pred = model_lookup.get(model_key, {})
        
        # Calculate implied probabilities
        imp_win = _imp_from_fixed(win_fixed)
        
        # Extract weight information (handle both dict and string formats)
        weight_info = runner.get("weight", "")
        if isinstance(weight_info, dict):
            weight = weight_info.get("allocated") or weight_info.get("total") or ""
        else:
            weight = str(weight_info) if weight_info else ""
        
        # Extract form information
        form = runner.get("last_twenty_starts", "")
        form_indicators = runner.get("form_indicators", [])
        
        # Extract speedmap information
        speedmap_info = runner.get("speedmap", {})
        speedmap_label = ""
        if isinstance(speedmap_info, dict):
            speedmap_label = (speedmap_info.get("label") or "").strip()
        
        # Extract edge information from form indicators
        edge_tags = []
        if isinstance(form_indicators, list):
            for indicator in form_indicators:
                group = (indicator.get("group") or "").strip()
                name = (indicator.get("name") or "").strip().lower()
                negative = bool(indicator.get("negative"))
                
                if group in {"Track_Distance", "Course_Distance"} and not negative:
                    edge_tags.append("TD+")
                if group == "Track" and not negative:
                    edge_tags.append("T+")
                if group == "Distance" and not negative:
                    edge_tags.append("D+")
                if "hat-trick" in name and not negative:
                    edge_tags.append("HTR")
        
        # Remove duplicates and join edge tags
        edge = " ".join(list(dict.fromkeys(edge_tags)))  # dict.fromkeys preserves order
        
        processed_runner = {
            'number': runner_number,
            'name': runner_name,
            'jockey': runner.get("jockey", ""),
            'barrier': runner.get("barrier", ""),
            'weight': weight,
            'form': form,
            'speedmap': speedmap_label,
            'edge': edge,
            'odds': {
                'win_fixed': win_fixed,
                'place_fixed': place_fixed,
                'win_tote': win_tote,
                'place_tote': place_tote
            },
            'implied_win_prob': imp_win,
            'model_prediction': model_pred
        }
        
        processed_runners.append(processed_runner)
    
    return {
        'race_info': race_info,
        'runners': processed_runners,
        'field_size': len(processed_runners)
    }

@app.route('/api/results/<date_str>/<int:meet_no>/<int:race_no>')
def get_race_results(date_str, meet_no, race_no):
    """Get race results for a completed race"""
    try:
        # Try to fetch results from local data first
        results_data = data_service.fetch_race_results(date_str, meet_no, race_no)
        
        if results_data:
            return jsonify({
                'success': True,
                'results': results_data,
                'race_completed': True
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Results not available',
                'race_completed': False
            })
            
    except Exception as e:
        logger.error(f"Error fetching results: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'race_completed': False
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Development server
    app.run(debug=True, host='0.0.0.0', port=8080)
