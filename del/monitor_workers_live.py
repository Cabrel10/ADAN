#!/usr/bin/env python3
"""
Real-time worker performance monitoring script
Displays live progress of all 4 workers
"""

import os
import re
from pathlib import Path
from datetime import datetime
import time

def get_worker_progress(worker_name):
    """Get current progress for a worker"""
    checkpoint_dir = f"/mnt/new_data/t10_training/checkpoints/{worker_name}"
    
    try:
        # Find latest checkpoint
        checkpoints = list(Path(checkpoint_dir).glob(f"{worker_name}_model_*.zip"))
        if not checkpoints:
            return None
        
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        # Extract steps from filename
        match = re.search(r'(\d+)_steps', latest.name)
        if not match:
            return None
        
        steps = int(match.group(1))
        size = latest.stat().st_size / (1024 * 1024)  # MB
        mtime = datetime.fromtimestamp(latest.stat().st_mtime)
        
        return {
            'steps': steps,
            'size': size,
            'updated': mtime,
            'path': str(latest)
   