import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# List of summary stats
dataset_summary = []

def record_image_summary(sub_dir: str, img_id: str, history_df: pd.DataFrame):
    """Extracts best metrics from a run and records them into dataset_summary."""
    if history_df.empty:
        return
        
    # Get best row based on peak PSNR
    best_row = history_df.loc[history_df['psnr'].idxmax()]
    
    summary_entry = {
        "difficulty_set": sub_dir,
        "image_id": img_id,
        "best_epoch": int(best_row['epoch']),
        "final_cost": float(history_df.iloc[-1]['cost']),
        "best_psnr": float(best_row['psnr']),
        "best_ssim": float(best_row['ssim']),
        "best_mse": float(best_row['mse']),
    }
    dataset_summary.append(summary_entry)


def generate_final_report(output_base: Path):
    """Computes category averages, outputs summary CSV, and generates portfolio plots."""
    if not dataset_summary:
        print("No runs recorded to summarize.")
        return

    # Convert recorded runs into DataFrame
    df = pd.DataFrame(dataset_summary)
    
    # Save per-image results
    df.to_csv(output_base / "all_images_detailed.csv", index=False)

    # Calculate Averages grouped by Difficulty Level
    grouped_avg = df.groupby("difficulty_set")[["best_psnr", "best_ssim", "best_mse", "final_cost"]].mean().reset_index()
    grouped_avg.rename(columns={
        "best_psnr": "Mean Peak PSNR (dB)",
        "best_ssim": "Mean Peak SSIM",
        "best_mse": "Mean MSE",
        "final_cost": "Mean Final Cost"
    }, inplace=True)

    # Save CSV
    grouped_avg.to_csv(output_base / "difficulty_summary_averages.csv", index=False)
    print("\n" + "="*50)
    print("DIV2K BENCHMARK SUMMARY AVERAGES:")
    print("="*50)
    print(grouped_avg.to_string(index=False))

    # Plot Summary Bar Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # PSNR
    ax1.bar(grouped_avg['difficulty_set'], grouped_avg['Mean Peak PSNR (dB)'], color=['#2ca02c', '#1f77b4', '#ff7f0e'])
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("Average Peak PSNR by Difficulty")
    ax1.tick_params(axis='x', rotation=15)

    # SSIM
    ax2.bar(grouped_avg['difficulty_set'], grouped_avg['Mean Peak SSIM'], color=['#2ca02c', '#1f77b4', '#ff7f0e'])
    ax2.set_ylabel("SSIM Score")
    ax2.set_title("Average Peak SSIM by Difficulty")
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(axis='x', rotation=15)

    plt.suptitle("DIV2K INR Benchmark Results Across Difficulty Levels", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_base / "div2k_summary_barplot.png", dpi=300)
    plt.close()