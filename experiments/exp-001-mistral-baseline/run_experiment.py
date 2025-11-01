#!/usr/bin/env python3
"""
Run script for Baseline Mistral Misinformation Detection Experiment
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run the baseline experiment"""
    
    # Get the project root directory
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # Path to the batch processing script
    batch_script = project_root / "scripts" / "run_mistral_batch.py"
    
    # Path to this experiment's config
    config_file = Path(__file__).parent / "config.yaml"
    
    print("=" * 60)
    print("Baseline Mistral Misinformation Detection Experiment")
    print("=" * 60)
    print(f"Project Root: {project_root}")
    print(f"Config File: {config_file}")
    print(f"Batch Script: {batch_script}")
    print()
    
    # Check if required files exist
    if not batch_script.exists():
        print(f"ERROR: Batch script not found at {batch_script}")
        return 1
        
    if not config_file.exists():
        print(f"ERROR: Config file not found at {config_file}")
        return 1
    
    # Run the experiment
    try:
        print("Starting baseline experiment...")
        print("This may take a while depending on the number of video files...")
        print()
        
        cmd = [sys.executable, str(batch_script), "--config", str(config_file)]
        result = subprocess.run(cmd, cwd=project_root)
        
        if result.returncode == 0:
            print()
            print("✅ Experiment completed successfully!")
            print(f"Results saved to: experiments/exp-001-mistral-baseline/results.csv")
            print()
            print("Next steps:")
            print("1. Run analysis: python scripts/analyze_experiment.py --exp-dir experiments/exp-001-mistral-baseline --top-k 20")
            print("2. Review results and compare with other experiments")
        else:
            print("\n❌ Experiment failed!")
            return result.returncode
            
    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error running experiment: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())