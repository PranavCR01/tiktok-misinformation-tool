#!/usr/bin/env python3
"""
Run the Chain of Thought (COT) experiment for misinformation detection.

This script runs the COT experiment and generates analysis plots.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    experiment_name = "exp-002-mistral-cot"
    config_path = f"experiments/{experiment_name}/config.yaml"
    
    print(f"🚀 Starting Chain of Thought (COT) experiment: {experiment_name}")
    print(f"📁 Working directory: {project_root}")
    print(f"⚙️  Config: {config_path}")
    print()
    
    # Check if config exists
    if not Path(config_path).exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    # Check if data directory exists
    data_dir = Path("data/videos")
    if not data_dir.exists():
        print(f"⚠️  Warning: Data directory not found: {data_dir}")
        print("   Make sure you have video files in the data/videos directory")
    
    try:
        # Run the batch processing
        print("🔄 Running batch processing with COT prompt...")
        cmd = [
            sys.executable, "scripts/run_mistral_batch.py",
            "--config", config_path
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Batch processing completed successfully")
        
        # Run the analysis
        print("📊 Generating analysis and plots...")
        cmd = [
            sys.executable, "scripts/analyze_experiment.py",
            "--exp-dir", f"experiments/{experiment_name}",
            "--top-k", "20"
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Analysis completed successfully")
        
        print()
        print("🎉 Experiment completed!")
        print(f"📂 Results available in: experiments/{experiment_name}/")
        print("📋 Files generated:")
        print("   - results.csv (raw model outputs)")
        print("   - README.md (analysis report)")  
        print("   - *.png (visualization plots)")
        print("   - keywords_top.csv (keyword analysis)")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running experiment: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Experiment interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()