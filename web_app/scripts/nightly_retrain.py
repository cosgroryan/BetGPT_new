#!/usr/bin/env python3
"""
Nightly Model Retraining Script
Safely retrains the model without affecting the running web app
"""

import os
import sys
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

# Setup logging
# Use a writable log location (inside container or temp directory)
log_file = '/var/log/betgpt_retrain.log'
try:
    # Try to create the log file to test permissions
    with open(log_file, 'a'):
        pass
except PermissionError:
    # Fall back to a writable location
    log_file = '/tmp/betgpt_retrain.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_project_root():
    """Get the project root directory"""
    # When running in Docker, we're in /app, so project root is /app
    # When running locally, we need to go up to the parent directory
    current_path = Path(__file__).parent
    
    # Check if we're in Docker by looking for /app directory
    if Path('/app').exists():
        return Path('/app')  # Docker environment
    else:
        # Local environment - go up from web_app/scripts to BetGPT_new
        return current_path.parent.parent  # web_app/scripts -> web_app -> BetGPT_new

def backup_current_model():
    """Create a backup of the current model"""
    project_root = get_project_root()
    artifacts_dir = project_root / 'artifacts'
    backup_dir = project_root / 'model_backups'
    
    # Create backup directory if it doesn't exist
    backup_dir.mkdir(exist_ok=True)
    
    # Create timestamped backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = backup_dir / f'model_backup_{timestamp}'
    
    if artifacts_dir.exists():
        shutil.copytree(artifacts_dir, backup_path)
        logger.info(f"Model backed up to: {backup_path}")
        return backup_path
    else:
        logger.warning("No existing model found to backup")
        return None

def update_dataset():
    """Update the dataset with latest race data"""
    logger.info("Updating dataset...")
    
    project_root = get_project_root()
    logger.info(f"Project root: {project_root}")
    
    try:
        # Change to project root for data update
        original_cwd = os.getcwd()
        os.chdir(project_root)
        logger.info(f"Changed to directory: {os.getcwd()}")
        
        # Run the data update script (without retraining)
        # Determine the correct script path based on environment
        if Path('/app').exists():
            # Docker environment
            script_path = 'scripts/web_app_scripts/update_data_retrain_model.py'
        else:
            # Local environment
            script_path = 'web_app/scripts/web_app_scripts/update_data_retrain_model.py'
        
        logger.info(f"Running script: {script_path}")
        logger.info(f"Script exists: {Path(script_path).exists()}")
        
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
        
        if result.returncode == 0:
            logger.info("Dataset updated successfully")
            return True
        else:
            logger.error(f"Dataset update failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Dataset update timed out after 30 minutes")
        return False
    except Exception as e:
        logger.error(f"Failed to update dataset: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def train_new_model():
    """Train a new model in a temporary location"""
    logger.info("Training new model...")
    
    project_root = get_project_root()
    
    # Create temporary directory for new model
    temp_artifacts_dir = project_root / 'artifacts_temp'
    temp_artifacts_dir.mkdir(exist_ok=True)
    
    try:
        # Change to project root for training
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        # Run the training script with retrain flag
        # Determine the correct script path based on environment
        if Path('/app').exists():
            # Docker environment
            script_path = 'scripts/web_app_scripts/update_data_retrain_model.py'
        else:
            # Local environment
            script_path = 'web_app/scripts/web_app_scripts/update_data_retrain_model.py'
        
        result = subprocess.run([
            sys.executable, script_path, '--retrain'
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            logger.info("Model training completed successfully")
            # The training script updates the main artifacts directory
            # We'll copy it to our temp location for atomic swapping
            artifacts_dir = project_root / 'artifacts'
            if artifacts_dir.exists():
                # Copy current artifacts to temp location
                shutil.copytree(artifacts_dir, temp_artifacts_dir, dirs_exist_ok=True)
                return temp_artifacts_dir
            else:
                logger.error("Training completed but no artifacts found")
                return None
        else:
            logger.error(f"Model training failed: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error("Model training timed out after 1 hour")
        return None
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return None
    finally:
        os.chdir(original_cwd)

def atomic_model_swap(new_artifacts_dir):
    """Atomically swap the new model with the current one"""
    logger.info("Performing atomic model swap...")
    
    project_root = get_project_root()
    artifacts_dir = project_root / 'artifacts'
    temp_artifacts_dir = project_root / 'artifacts_temp'
    
    try:
        # Remove old artifacts directory
        if artifacts_dir.exists():
            shutil.rmtree(artifacts_dir)
        
        # Move new artifacts to production location
        shutil.move(str(temp_artifacts_dir), str(artifacts_dir))
        
        logger.info("Model swap completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Model swap failed: {e}")
        return False

def cleanup_old_backups():
    """Clean up old model backups (keep last 7 days)"""
    project_root = get_project_root()
    backup_dir = project_root / 'model_backups'
    
    if not backup_dir.exists():
        return
    
    # Remove backups older than 7 days
    cutoff_date = datetime.now().timestamp() - (7 * 24 * 60 * 60)
    
    for backup_path in backup_dir.iterdir():
        if backup_path.is_dir() and backup_path.stat().st_mtime < cutoff_date:
            shutil.rmtree(backup_path)
            logger.info(f"Removed old backup: {backup_path}")

def main():
    """Main retraining process"""
    logger.info("Starting nightly model retraining...")
    
    try:
        # Step 1: Backup current model
        backup_path = backup_current_model()
        
        # Step 2: Update dataset
        if not update_dataset():
            logger.error("Dataset update failed, aborting retraining")
            return False
        
        # Step 3: Train new model
        new_model_dir = train_new_model()
        if not new_model_dir:
            logger.error("Model training failed, aborting")
            return False
        
        # Step 4: Atomic swap
        if not atomic_model_swap(new_model_dir):
            logger.error("Model swap failed")
            return False
        
        # Step 5: Cleanup old backups
        cleanup_old_backups()
        
        logger.info("Nightly model retraining completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Retraining process failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
