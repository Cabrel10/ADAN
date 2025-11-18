#!/usr/bin/env python3
"""
Generate 5 independent Colab notebooks for parallel training.
Each notebook has its own checkpoint directory and logs.
"""

import json
from pathlib import Path

BASE_NOTEBOOK = Path("ADAN_Colab_Training_V1.ipynb")

# Configuration for each notebook version
VERSIONS = {
    "V2": {
        "checkpoint_dir": "checkpoints_v2",
        "log_dir": "logs_v2",
        "description": "Instance 2 of 5 - Independent Training Session",
        "worker_seed": 42,
    },
    "V3": {
        "checkpoint_dir": "checkpoints_v3",
        "log_dir": "logs_v3",
        "description": "Instance 3 of 5 - Independent Training Session",
        "worker_seed": 123,
    },
    "V4": {
        "checkpoint_dir": "checkpoints_v4",
        "log_dir": "logs_v4",
        "description": "Instance 4 of 5 - Independent Training Session",
        "worker_seed": 456,
    },
    "V5": {
        "checkpoint_dir": "checkpoints_v5",
        "log_dir": "logs_v5",
        "description": "Instance 5 of 5 - Independent Training Session",
        "worker_seed": 789,
    },
}


def generate_notebook(version: str, config: dict) -> None:
    """Generate a notebook for a specific version."""
    
    # Load base notebook
    with open(BASE_NOTEBOOK, 'r') as f:
        notebook = json.load(f)
    
    # Update title and description
    notebook['cells'][0]['source'] = [
        f"# 🚀 ADAN Trading Bot - Colab Training {version}\n",
        f"**{config['description']}**\n",
        "\n",
        "This notebook runs independently. If it fails, you can restart with another version."
    ]
    
    # Update checkpoint and log directories in training phase
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Replace checkpoint and log directories
            source = source.replace('checkpoints_v1', config['checkpoint_dir'])
            source = source.replace('logs_v1', config['log_dir'])
            
            # Update instance reference
            source = source.replace('Instance: V1', f'Instance: {version}')
            
            cell['source'] = source.split('\n')
            # Fix the split to maintain proper formatting
            cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line 
                            for i, line in enumerate(cell['source'])]
    
    # Save new notebook
    output_file = Path(f"ADAN_Colab_Training_{version}.ipynb")
    with open(output_file, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✅ Generated {output_file}")


def main():
    """Generate all notebooks."""
    print("🔧 Generating 5 independent Colab notebooks...\n")
    
    for version, config in VERSIONS.items():
        generate_notebook(version, config)
    
    print("\n✅ All notebooks generated successfully!")
    print("\n📊 Summary:")
    print("  V1: checkpoints_v1, logs_v1")
    print("  V2: checkpoints_v2, logs_v2")
    print("  V3: checkpoints_v3, logs_v3")
    print("  V4: checkpoints_v4, logs_v4")
    print("  V5: checkpoints_v5, logs_v5")
    print("\n🚀 Each notebook is independent and can run in parallel!")


if __name__ == "__main__":
    main()
