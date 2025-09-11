#!/usr/bin/env python3
"""
Test script for the nightly retraining process
Run this to test the retraining without waiting for cron
"""

import sys
import os
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import numpy
        import pandas
        import sklearn
        return True
    except ImportError:
        return False

def run_docker_test():
    """Run the test using Docker"""
    print("Dependencies not available on host, running in Docker...")
    
    # Change to web_app directory
    web_app_dir = Path(__file__).parent.parent
    os.chdir(web_app_dir)
    
    # Run the Docker test script
    result = subprocess.run(['./scripts/test_retrain_docker.sh'], 
                          capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

def run_local_test():
    """Run the test locally"""
    print("Running local test...")
    
    # Add the scripts directory to Python path
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))

    from nightly_retrain import main
    return main()

if __name__ == "__main__":
    print("Testing nightly model retraining...")
    
    if check_dependencies():
        success = run_local_test()
    else:
        success = run_docker_test()
    
    if success:
        print("✅ Test retraining completed successfully!")
    else:
        print("❌ Test retraining failed!")
        sys.exit(1)