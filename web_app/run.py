#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BetGPT Web App Startup Script
Run this to start the web application.
"""

import os
import sys
from app import app

if __name__ == '__main__':
    # Set environment variables for development
    os.environ.setdefault('FLASK_ENV', 'development')
    os.environ.setdefault('FLASK_DEBUG', '1')
    
    # Run the application
    print("Starting BetGPT Web App...")
    print("Access the application at: http://localhost:8080")
    print("Press Ctrl+C to stop the server")
    
    app.run(
        host='0.0.0.0',
        port=8080,
        debug=True,
        use_reloader=False
    )
