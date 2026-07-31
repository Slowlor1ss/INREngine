import subprocess
import time
import random
import csv
import pyautogui
import re
import os
import argparse

# 1. SETUP PYTHON ARGPARSE FOR THE SWEEP SCRIPT
# This creates a user-friendly CLI to control the automation script itself.
parser = argparse.ArgumentParser(description='Automated Hyperparameter Sweep for C++ SIREN')
parser.add_argument('--tests', dest='tests', type=int, default=9999, help='Number of random tests to run')
parser.add_argument('--duration', dest='duration', type=int, default=1800, help='Duration per test in seconds')
parser.add_argument('--img', dest='img', type=str, default='W:\\Other\\Personal\\3b1b_backprop\\Training_Data\\camera.bmp', help='Target image to train on')
args = parser.parse_args()

PROGRAM_PATH = "W:\\Other\\Personal\\3b1b_backprop\\x64\\Release\\3b1b_backprop.exe" 
CSV_FILE = "master_sweep.csv"
TEMP_LOG = "temp_log.txt"

# 2. DEFINE THE HYPERPARAMETER SEARCH SPACE
# The script will randomly pick one value from these arrays for every test
SPACE_WIDTHS = [64]#[16, 32, 64, 128]
SPACE_DEPTHS = [3]#, 4, 5, 6]
SPACE_PE = [1]
SPACE_FREQ = [2, 5, 7, 10, 13, 15]
SPACE_BATCH = [32]
SPACE_LR = [0.15, 0.01, 0.001, 0.0005, 0.0001]

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Test_ID", "Width", "Depth", "PE_Enabled", "PE_Freq", "Batch_Size", "Learning_Rate", "Final_Cost"])

print(f"🚀 Starting Sweep: Running {args.tests} tests for {args.duration}s each.")

for test_id in range(1, args.tests + 1):
    
    # Randomly select hyperparameters for this run
    width = random.choice(SPACE_WIDTHS)
    depth = random.choice(SPACE_DEPTHS)
    layers = [width] * depth + [3]
    layers_str = list(map(str, layers))
    
    use_pe = random.choice(SPACE_PE)
    freq = random.choice(SPACE_FREQ) if use_pe == 1 else 0
    batch = random.choice(SPACE_BATCH)
    lr = random.choice(SPACE_LR)
    
    # Construct the C++ command
    cmd = [
        PROGRAM_PATH,
        "--i", args.img,
        "--o", f"SavedWeights/weights_test_{test_id}.csv",
        "--set-pe", str(use_pe),
        "--freq", str(freq),
        "--batch", str(batch),
        "--lr", str(lr),
        "--set-live", "1", # Keep at 1 so the window opens for pyautogui
        "--layers"
    ] + layers_str
    
    print(f"\n==================================================")
    print(f"Starting Test {test_id}/{args.tests}")
    print(f"Config: PE={use_pe} (Freq:{freq}), Batch={batch}, LR={lr}, Layers={width}x{depth}")
    print(f"==================================================")

    with open(TEMP_LOG, "w") as log_file:
        process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        
        start_time = time.time()
        while time.time() - start_time < args.duration:
            elapsed = time.time() - start_time
            rem = int(args.duration - elapsed)
            print(f"\r⏳ Time remaining: {rem // 60}m {rem % 60}s...", end="")
            time.sleep(1)
            
            if process.poll() is not None:
                print("\nProcess exited prematurely!")
                break
                
    print("\nSaving weights and terminating...")
    pyautogui.press('w')
    time.sleep(2) # Buffer to allow C++ to write to disk
    
    process.terminate()
    process.wait()

    # Parse the final cost
    final_cost = "NaN"
    if os.path.exists(TEMP_LOG):
        with open(TEMP_LOG, "r") as log_file:
            content = log_file.read()
            matches = re.findall(r'(?:loss|cost).*?([0-9]*\.?[0-9]+)', content, re.IGNORECASE)
            if matches:
                final_cost = matches[-1] 

    print(f"Test {test_id} Complete! Final Cost: {final_cost}")

    # Append to master CSV
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([test_id, width, depth, use_pe, freq, batch, lr, final_cost])