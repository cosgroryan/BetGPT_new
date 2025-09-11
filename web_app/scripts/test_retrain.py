#!/usr/bin/env python3
"""
Test script for the nightly retraining process
Run this to test the retraining without waiting for cron
"""

import sys
import os
from pathlib import Path

# Add the scripts directory to Python path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from nightly_retrain import main

if __name__ == "__main__":
    print("Testing nightly model retraining...")
    success = main()
    
    if success:
        print("✅ Test retraining completed successfully!")
    else:
        print("❌ Test retraining failed!")
        sys.exit(1)