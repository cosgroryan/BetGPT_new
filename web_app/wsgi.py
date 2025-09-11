#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WSGI entry point for production deployment
"""

import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
