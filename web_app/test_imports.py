#!/usr/bin/env python3
"""
Test script to verify Python imports work correctly in both local and Docker environments.
Run this to debug import issues before rebuilding containers.
"""

import os
import sys

print("🔍 Testing Python Import Paths...")
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {__file__}")

# Add paths for imports - handle both local and Docker environments
current_dir = os.path.dirname(os.path.abspath(__file__))

# In Docker container, scripts are mounted at /app/scripts
# In local development, scripts are in web_app/scripts
if os.path.exists('/app/scripts'):
    # Docker environment
    print("🐳 Docker environment detected")
    sys.path.append('/app/scripts/web_app_scripts')
    sys.path.append('/app/scripts/helper_scripts')
    parent_dir = '/app'
else:
    # Local development environment
    print("💻 Local development environment detected")
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    sys.path.append(os.path.join(current_dir, 'scripts', 'web_app_scripts'))
    sys.path.append(os.path.join(current_dir, 'scripts', 'helper_scripts'))

print(f"Python path: {sys.path}")

# Test imports
try:
    print("📦 Testing imports...")
    from recommend_picks_NN import model_win_table
    print("✅ recommend_picks_NN imported successfully")
    
    from pytorch_pre import load_model_and_predict
    print("✅ pytorch_pre imported successfully")
    
    print("🎉 All imports successful!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Available files in scripts directory:")
    scripts_dir = '/app/scripts' if os.path.exists('/app/scripts') else os.path.join(current_dir, 'scripts')
    if os.path.exists(scripts_dir):
        for root, dirs, files in os.walk(scripts_dir):
            for file in files:
                print(f"  {os.path.join(root, file)}")
    else:
        print(f"  Scripts directory not found: {scripts_dir}")
    sys.exit(1)
