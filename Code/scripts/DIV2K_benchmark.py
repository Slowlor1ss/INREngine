import subprocess
import time
import imageio.v3 as iio
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Run in headless mode
import matplotlib.pyplot as plt
import numpy as np
import sys

from pathlib import Path
from matplotlib.ticker import ScalarFormatter

# Dynamically resolve the absolute path
viz_dir = Path(__file__).parent / "viz"
sys.path.append(str(viz_dir.resolve()))
from shared_memory_bridge import SharedMemoryBridge
from metrics import compute_image_metrics
from summary_metrics_DIV2K import generate_final_report, record_image_summary

# Note: Because standard INRs perform single instance optimization per signal 
# f(x,y) to (R,G,B), DIV2K is used here as a diverse evaluation benchmark 
# rather than a generalization training set (unlike feedforward CNNs/Transformers)
def run_div2k_suite(div2k_base: str, exe_path: str, output_base: str):
    base_path = Path(div2k_base)
    out_path = Path(output_base)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Directories to scan
    sub_dirs = ["DIV2K_train_HR", "DIV2K_train_LR", "DIV2K_train_LR_mild"]
    
    for sub in sub_dirs:
        dir_path = base_path / sub
        if not dir_path.exists(): continue
            
        for img_file in sorted(dir_path.glob("*.png")):#[:3]:
            run_name = f"{sub}_{img_file.stem}"
            run_out = out_path / run_name
            run_out.mkdir(exist_ok=True)
            
            img_id = img_file.name[:4]
            hr_path = base_path / "DIV2K_train_HR" / f"{img_id}.png"
            cmd = [
                exe_path, 
                "--i", str(img_file), 
                "--o", str(run_out),
                "--HDin", str(hr_path), 
                "--set-live", "1",
                "--benchmark", "1",
                "--denoise", "1" if sub != "DIV2K_train_HR" and sub != "DIV2K_train_LR" else "0",
                "--scale", "4.0" if sub != "DIV2K_train_HR" else "1.0"
            ]
            proc = subprocess.Popen(cmd)
            
            # Hook into Shared Memory
            bridge = SharedMemoryBridge(img_file.stem) # "INR_Default"
            if not bridge.wait_for_producer(timeout_s=15.0, poll_s=0.0001):
                print(f"Failed to connect to {run_name}")
                proc.terminate()
                continue
            else:
                print(f"Py - C++ connection success for {run_name}!\n")
                
            history, frames = [], []
            last_frame_raw = None
            target = bridge.read_target_frame()
            
            history, frames = [], []
            target = bridge.read_target_frame()
            
            # Keep track of the last frame we processed
            last_seen_frame = -1
            while bridge.read_stats().running:
                current_frame_count = bridge.last_frame_counter()
                if current_frame_count > last_seen_frame:
                    frame = bridge.read_current_frame()
                    stats = bridge.read_stats()
                    
                    if frame is not None and target is not None:
                        print(f"Caught: {current_frame_count}")
                        # Calculate metrics dynamically
                        mets = compute_image_metrics(frame, target, stats.epoch, stats.cost)
                        history.append(mets)

                        # Just store the reference
                        last_frame_raw = frame
                        
                        # Store frame for GIF (convert float [0,1] to uint8)
                        # frames.append((np.clip(frame, 0, 1) * 255).astype(np.uint8))
                        
                        # Update our tracker so we don't process this frame again
                        last_seen_frame = current_frame_count
                        
                #time.sleep(0.5) 
            
            proc.wait()

            # Save artifacts for this specific image run
            if last_frame_raw is not None:
                final_img = (np.clip(last_frame_raw, 0, 1) * 255).astype(np.uint8)
                iio.imwrite(run_out / f"{run_name}_final.png", final_img)
            
            if frames:
                iio.imwrite(run_out / f"{run_name}.gif", frames, duration=1000/25, loop=0)
                
            if history:
                df = pd.DataFrame([vars(h) for h in history])
                df.to_csv(run_out / "metrics.csv", index=False)
                plot_run_metrics(df, run_out / f"{run_name}_graph.png")
                
                # Log individual run's peak stats into memory
                record_image_summary(sub, img_id, df)

    # Generate dataset-wide averages, summary CSV, and plot
    generate_final_report(out_path)

def plot_run_metrics(df, out_path):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.plot(df['epoch'], df['ssim'], color='tab:red', label='SSIM')
    ax1.set_xlabel('Epoch (Linear-Log Blend)')
    ax1.set_ylabel('SSIM', color='tab:red')
    ax1.invert_yaxis()
    
    # linthresh=300 means epochs 0 to 300 are plotted normally (preserving the curve),
    # while everything after 300 gets compressed logarithmically.
    ax1.set_xscale('symlog', linthresh=300) 
    ax1.set_xticks([0, 25, 50, 75, 100, 135, 175, 225, 275, 350, 500, 750, 1250, 2000, 3500])
    
    formatter = ScalarFormatter()
    formatter.set_scientific(False) # Forces plain numbers like 10, 100, 1000
    ax1.xaxis.set_major_formatter(formatter)
    
    ax2 = ax1.twinx()
    ax2.plot(df['epoch'], df['psnr'], color='tab:blue', label='PSNR')
    ax2.set_ylabel('PSNR (dB)', color='tab:blue')
    
    plt.title('Training Metrics')
    fig.tight_layout()
    plt.savefig(out_path)
    plt.close()

if __name__ == "__main__":
    run_div2k_suite("../../Training_Data", "../../x64/Training/3b1b_backprop.exe", "../../OUT/Portfolio_Output")