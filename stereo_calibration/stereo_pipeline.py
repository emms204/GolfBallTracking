#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import subprocess
import time

def run_command(command, desc):
    """
    Run a command with a description.
    """
    print(f"\n{'='*80}")
    print(f"  {desc}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(command)}")
    
    start_time = time.time()
    
    # Run the command and capture output
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Print output in real-time
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
    
    process.wait()
    
    elapsed_time = time.time() - start_time
    
    # Check if the command succeeded
    if process.returncode == 0:
        print(f"\nCommand completed successfully in {elapsed_time:.2f} seconds.")
        return True
    else:
        print(f"\nCommand failed with return code {process.returncode} after {elapsed_time:.2f} seconds.")
        return False

def create_directories():
    """
    Create output directories.
    """
    Path("calibration_results").mkdir(exist_ok=True)
    Path("visualization").mkdir(exist_ok=True)
    Path("reconstruction").mkdir(exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description='Run the complete stereo calibration and reconstruction pipeline')
    parser.add_argument('--skip_calibration', action='store_true', help='Skip the calibration step')
    parser.add_argument('--skip_visualization', action='store_true', help='Skip the visualization step')
    parser.add_argument('--skip_reconstruction', action='store_true', help='Skip the reconstruction step')
    parser.add_argument('--master_img', type=str, default='../Chess 8mm/Master/1.jpeg', help='Path to master camera image for visualization/reconstruction')
    parser.add_argument('--slave_img', type=str, default='../Chess 8mm/Slave/1.jpeg', help='Path to slave camera image for visualization/reconstruction')
    
    args = parser.parse_args()
    
    # Create output directories
    create_directories()
    
    steps_completed = 0
    steps_failed = 0
    
    # Step 1: Calibration
    if not args.skip_calibration:
        print("\nStep 1: Running Stereo Calibration")
        if run_command(["python", "stereo_calibration.py"], "Camera Calibration"):
            steps_completed += 1
        else:
            steps_failed += 1
            print("WARNING: Calibration failed, but continuing with pipeline...")
    else:
        print("\nSkipping Calibration step (--skip_calibration flag is set)")
        
    # Step 2: Visualization
    if not args.skip_visualization:
        print("\nStep 2: Running Calibration Visualization")
        if run_command([
            "python", "visualize_calibration.py",
            "--master_img", args.master_img,
            "--slave_img", args.slave_img
        ], "Calibration Visualization"):
            steps_completed += 1
        else:
            steps_failed += 1
            print("WARNING: Visualization failed, but continuing with pipeline...")
    else:
        print("\nSkipping Visualization step (--skip_visualization flag is set)")
        
    # Step 3: Reconstruction
    if not args.skip_reconstruction:
        print("\nStep 3: Running 3D Reconstruction")
        if run_command([
            "python", "stereo_reconstruction.py",
            "--master_img", args.master_img,
            "--slave_img", args.slave_img
        ], "3D Reconstruction"):
            steps_completed += 1
        else:
            steps_failed += 1
    else:
        print("\nSkipping Reconstruction step (--skip_reconstruction flag is set)")
    
    # Print summary
    total_steps = 3 - args.skip_calibration - args.skip_visualization - args.skip_reconstruction
    print("\n" + "="*80)
    print(f"Pipeline Summary:")
    print(f"  {steps_completed} of {total_steps} steps completed successfully")
    if steps_failed > 0:
        print(f"  {steps_failed} steps failed")
    
    print("\nOutput directories:")
    print("  - calibration_results/: Contains calibration parameters")
    print("  - visualization/: Contains calibration visualization results")
    print("  - reconstruction/: Contains 3D reconstruction results")
    print("="*80)
    
    # Return success if all steps passed
    return steps_failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 