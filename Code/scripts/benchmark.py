import subprocess
import os
import shutil
import imageio.v3 as iio
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

runs = [
    {
        "name": "Siren_Base_2x256",
        "cmd": ["../../x64/Training/3b1b_backprop.exe", "--i", "../../Training_Data/0064_x4.bmp", "--o", "../../OUT/results/siren_base/0064_x4_siren_2x256.bmp", 
                "--layers", "256", "256", "3", "--act", "Siren", "Siren", "None", 
                "--batch", "65536", "--lr", "1", "--set-live", "0"]
    },
    {
        "name": "Wire_Base_2x256",
        "cmd": ["../../x64/Training/3b1b_backprop.exe", "--i", "../../Training_Data/0064_x4.bmp", "--o", "../../OUT/results/wire_base/0064_x4_wire_2x256.bmp", 
                "--layers", "256", "256", "3", "--act", "Wire", "Wire", "None", 
                "--batch", "65536", "--lr", "1", "--set-live", "0"]
    },
    {
        "name": "Siren_GPE_6x256_b8192",
        "cmd": ["../../x64/Training/3b1b_backprop.exe", "--i", "../../Training_Data/0064_x4.bmp", "--o", "../../OUT/results/siren_6x256_GPE_b8192/0064_x4_siren_GPE_6x256_b8192.bmp", 
                "--layers", "256", "256", "256", "256", "256", "256", "3", "--act", "Siren", "Siren", "Siren", "Siren", "Siren", "Siren", "None", 
                "--batch", "8192", "--lr", "1", "--set-live", "0", "--set-gaussian-pe", "1"]
    },
    {
        "name": "Siren_PE_6x256_b8192",
        "cmd": ["../../x64/Training/3b1b_backprop.exe", "--i", "../../Training_Data/0064_x4.bmp", "--o", "../../OUT/results/siren_6x256_PE_b8192/0064_x4_siren_PE_6x256_b8192.bmp", 
                "--layers", "256", "256", "256", "256", "256", "256", "3", "--act", "Siren", "Siren", "Siren", "Siren", "Siren", "Siren", "None", 
                "--batch", "8192", "--lr", "1", "--set-live", "0", "--set-pe", "1"]
    },
    {
        "name": "LeakyReLU_Base_2x256",
        "cmd": ["../../x64/Training/3b1b_backprop.exe", "--i", "../../Training_Data/0064_x4.bmp", "--o", "../../OUT/results/LeakyReLU_base/0064_x4_LeakyReLU_2x256.bmp", 
                "--layers", "256", "256", "3", "--act", "LeakyReLU", "LeakyReLU", "None", 
                "--batch", "65536", "--lr", "1", "--set-live", "0"]
    },
    {
        "name": "ReLU_Base_2x256",
        "cmd": ["../../x64/Training/3b1b_backprop.exe", "--i", "../../Training_Data/0064_x4.bmp", "--o", "../../OUT/results/ReLU_base/0064_x4_ReLU_2x256.bmp", 
                "--layers", "256", "256", "3", "--act", "ReLU", "ReLU", "None", 
                "--batch", "65536", "--lr", "1", "--set-live", "0"]
    },
    {
        "name": "Sigmoid_Base_2x256",
        "cmd": ["../../x64/Training/3b1b_backprop.exe", "--i", "../../Training_Data/0064_x4.bmp", "--o", "../../OUT/results/Sigmoid_base/0064_x4_Sigmoid_2x256.bmp", 
                "--layers", "256", "256", "3", "--act", "Sigmoid", "Sigmoid", "None", 
                "--batch", "65536", "--lr", "1", "--set-live", "0"]
    },
    {
        "name": "Tanh_Base_2x256",
        "cmd": ["../../x64/Training/3b1b_backprop.exe", "--i", "../../Training_Data/0064_x4.bmp", "--o", "../../OUT/results/Tanh_base/0064_x4_Tanh_2x256.bmp", 
                "--layers", "256", "256", "3", "--act", "Tanh", "Tanh", "None", 
                "--batch", "65536", "--lr", "1", "--set-live", "0"]
    },
    # --- CAMERA DATASET ---
    {
        "name": "Siren_Base_2x256_Cam",
        "cmd": ["../../x64/Training/3b1b_backprop.exe", "--i", "../../Training_Data/camera.bmp", "--o", "../../OUT/results/siren_base/camera_siren_2x256.bmp", 
                "--layers", "256", "256", "3", "--act", "Siren", "Siren", "None", 
                "--batch", "65536", "--lr", "1", "--set-live", "0"]
    },
    {
        "name": "Wire_Base_2x256_Cam",
        "cmd": ["../../x64/Training/3b1b_backprop.exe", "--i", "../../Training_Data/camera.bmp", "--o", "../../OUT/results/wire_base/camera_wire_2x256.bmp", 
                "--layers", "256", "256", "3", "--act", "Wire", "Wire", "None", 
                "--batch", "65536", "--lr", "1", "--set-live", "0"]
    },
    {
        "name": "Siren_GPE_6x256_b8192_Cam",
        "cmd": ["../../x64/Training/3b1b_backprop.exe", "--i", "../../Training_Data/camera.bmp", "--o", "../../OUT/results/siren_6x256_GPE_b8192/camera_siren_GPE_6x256_b8192.bmp", 
                "--layers", "256", "256", "256", "256", "256", "256", "3", "--act", "Siren", "Siren", "Siren", "Siren", "Siren", "Siren", "None", 
                "--batch", "8192", "--lr", "1", "--set-live", "0", "--set-gaussian-pe", "1"]
    },
    {
        "name": "Siren_PE_6x256_b8192_Cam",
        "cmd": ["../../x64/Training/3b1b_backprop.exe", "--i", "../../Training_Data/camera.bmp", "--o", "../../OUT/results/siren_6x256_PE_b8192/camera_siren_PE_6x256_b8192.bmp", 
                "--layers", "256", "256", "256", "256", "256", "256", "3", "--act", "Siren", "Siren", "Siren", "Siren", "Siren", "Siren", "None", 
                "--batch", "8192", "--lr", "1", "--set-live", "0", "--set-pe", "1"]
    },
    {
        "name": "LeakyReLU_Base_2x256_Cam",
        "cmd": ["../../x64/Training/3b1b_backprop.exe", "--i", "../../Training_Data/camera.bmp", "--o", "../../OUT/results/LeakyReLU_base/camera_LeakyReLU_2x256.bmp", 
                "--layers", "256", "256", "3", "--act", "LeakyReLU", "LeakyReLU", "None", 
                "--batch", "65536", "--lr", "1", "--set-live", "0"]
    },
    {
        "name": "ReLU_Base_2x256_Cam",
        "cmd": ["../../x64/Training/3b1b_backprop.exe", "--i", "../../Training_Data/camera.bmp", "--o", "../../OUT/results/ReLU_base/camera_ReLU_2x256.bmp", 
                "--layers", "256", "256", "3", "--act", "ReLU", "ReLU", "None", 
                "--batch", "65536", "--lr", "1", "--set-live", "0"]
    },
    {
        "name": "Sigmoid_Base_2x256_Cam",
        "cmd": ["../../x64/Training/3b1b_backprop.exe", "--i", "../../Training_Data/camera.bmp", "--o", "../../OUT/results/Sigmoid_base/camera_Sigmoid_2x256.bmp", 
                "--layers", "256", "256", "3", "--act", "Sigmoid", "Sigmoid", "None", 
                "--batch", "65536", "--lr", "1", "--set-live", "0"]
    },
    {
        "name": "Tanh_Base_2x256_Cam",
        "cmd": ["../../x64/Training/3b1b_backprop.exe", "--i", "../../Training_Data/camera.bmp", "--o", "../../OUT/results/Tanh_base/camera_Tanh_2x256.bmp", 
                "--layers", "256", "256", "3", "--act", "Tanh", "Tanh", "None", 
                "--batch", "65536", "--lr", "1", "--set-live", "0"]
    },
    # BONUS
        {
        "name": "Wire_Base_2x626_Cam",
        "cmd": ["../../x64/Training/3b1b_backprop.exe", "--i", "../../Training_Data/0064_x4.bmp", "--o", "../../OUT/results/wire_base/0064_x4_wire_2x626.bmp", 
                "--layers", "526", "526", "3", "--act", "Wire", "Wire", "None", 
                "--batch", "65536", "--lr", "1", "--set-live", "0"]
    },
        {
        "name": "Wire_Base_4x256_Cam",
        "cmd": ["../../x64/Training/3b1b_backprop.exe", "--i", "../../Training_Data/0064_x4.bmp", "--o", "../../OUT/results/wire_base/0064_x4_wire_4x256.bmp", 
                "--layers", "256", "256", "256", "256", "3", "--act", "Wire", "Wire", "Wire", "Wire", "None", 
                "--batch", "65536", "--lr", "0.1", "--set-live", "0"]
    }
]

def make_gif(image_folder, output_path, fps=25):
    """Stitches all BMPs in a folder into a GIF."""
    path = Path(image_folder)
    if not path.exists(): return
    
    # Sort files by batch number assuming format like 'step_0100.bmp'
    images = sorted(path.glob("*.bmp")) 
    frames = [iio.imread(img) for img in images]
    
    if frames:
        iio.imwrite(output_path, frames, duration=1000/fps, loop=0)
        print(f"Saved GIF: {output_path}")

def plot_metrics(csv_file, output_path):
    """Reads the C++ CSV and plots Cost and PSNR."""
    if not os.path.exists(csv_file): return
    
    df = pd.read_csv(csv_file)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot Cost
    color = 'tab:red'
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss (MSE/L1)', color=color)
    ax1.plot(df['Batch'], df['Cost'], color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Plot PSNR on same graph, different axis
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('PSNR (dB)', color=color)
    ax2.plot(df['Batch'], df['PSNR'], color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Training Metrics')
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()

def extract_best_images(csv_file, rgb_folder, grad_folder, output_rgb_path, output_grad_path):
    """Finds the highest PSNR in the CSV and copies the corresponding RGB and Grad images."""
    if not os.path.exists(csv_file): return
    
    df = pd.read_csv(csv_file)
    if df.empty: return

    # Find the exact row index where PSNR is the highest
    best_idx = df['PSNR'].idxmax()
    best_batch = int(df.loc[best_idx, 'Batch'])
    best_psnr = float(df.loc[best_idx, 'PSNR'])
    
    # Reconstruct the C++ filename (e.g., step_01230.bmp)
    best_image_name = f"step_{best_batch:05d}.bmp"
    
    # Copy RGB
    best_rgb_src = Path(rgb_folder) / best_image_name
    if best_rgb_src.exists():
        shutil.copy2(best_rgb_src, output_rgb_path)
        
    # Copy Gradient
    best_grad_src = Path(grad_folder) / best_image_name
    if best_grad_src.exists():
        shutil.copy2(best_grad_src, output_grad_path)
        
    print(f"Extracted Best Images (Batch {best_batch} @ {best_psnr:.2f} dB)")

# --- Execute the Pipeline ---
for run in runs:
    print(f"\n{'='*40}")
    print(f"Starting Run: {run['name']}")
    print(f"{'='*40}")

    cmd = run['cmd']
    
    # Run the C++ Engine (Wait for it to finish)
    subprocess.run(cmd)
    
    # Dynamically parse the "--o" path from the command array
    try:
        o_index = cmd.index("--o")
        output_full_path = Path(cmd[o_index + 1])
    except ValueError:
        print(f"Warning: Could not find '--o' in cmd for {run['name']}. Skipping output parsing.")
        continue

    # Replicate the C++ folder creation logic
    base_path = output_full_path.parent
    stem = output_full_path.stem
    bench_folder = base_path / f"benchmark_{stem}"
    
    # Process the Outputs
    make_gif(bench_folder / "rgb", bench_folder / f"{run['name']}_rgb.gif")
    make_gif(bench_folder / "grad", bench_folder / f"{run['name']}_grad.gif")
    plot_metrics(bench_folder / "metrics.csv", bench_folder / f"{run['name']}_graph.png")
    extract_best_images(
        bench_folder / "metrics.csv", 
        bench_folder / "rgb", 
        bench_folder / "grad", 
        bench_folder / f"{run['name']}_best_rgb.bmp",
        bench_folder / f"{run['name']}_best_grad.bmp"
    )

print("\nAll benchmarking complete!")