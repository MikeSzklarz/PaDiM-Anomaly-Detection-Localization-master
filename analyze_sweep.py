import os
import glob
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_summary_chart(summary_df, save_path):
    """
    Creates and saves a grouped bar chart of the IoU scores.
    """
    # We need the parameter to annotate the plot
    df_for_plot = summary_df[['Best_NP_IoU', 'ClassF1_IoU', 'Best_NP_Param (%)']].copy()
    
    # Rename columns for a prettier legend
    df_for_plot.rename(columns={
        'Best_NP_IoU': 'Best NormalPercentile',
        'ClassF1_IoU': 'ClassF1 (Original)'
    }, inplace=True)

    # Use pd.melt() to convert from wide-form to long-form data
    df_melted = df_for_plot.reset_index().melt(
        id_vars=['Class', 'Best_NP_Param (%)'], # Keep param for annotation
        var_name='Method', 
        value_name='IoU'
    )
    
    # Ensure IoU is numeric
    df_melted['IoU'] = pd.to_numeric(df_melted['IoU'])

    # Create the plot
    plt.figure(figsize=(16, 9))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(
        x='Class', 
        y='IoU', 
        hue='Method', 
        data=df_melted,
        palette='viridis' # Use a colorblind-friendly palette
    )
    
    # Set titles and labels
    ax.set_title(
        'Segmentation Performance: Best NormalPercentile vs. ClassF1 (Original)',
        fontsize=18,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('MVTec Class', fontsize=14, labelpad=10)
    ax.set_ylabel('Mean IoU (on Anomalous Images)', fontsize=14, labelpad=10)
    
    # Improve readability
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylim(0, 1.0) # IoU is 0-1
    ax.legend(title='Thresholding Method', fontsize=12, title_fontsize=13, loc='upper left')
    
    # Add value labels on top of each bar
    # We need to get the param data for the 'Best NormalPercentile' bars
    num_classes = len(df_melted['Class'].unique())
    
    # This logic assumes Seaborn plots all 'Best NormalPercentile' bars first,
    # then all 'ClassF1 (Original)' bars.
    param_data = df_melted[
        df_melted['Method'] == 'Best NormalPercentile'
    ]['Best_NP_Param (%)'].values

    for i, p in enumerate(ax.patches):
        height = p.get_height()
        # Basic text is just the IoU score
        text = f'{height:.3f}'
        
        # Check if this is one of the first 'num_classes' bars (Best NP)
        if i < num_classes:
            try:
                param = param_data[i % num_classes]
                # Add the parameter to the text
                text += f'\n(@{param:.1f}%)'
            except IndexError:
                pass # Fail silently if index is out of bounds
        
        ax.annotate(
            text,
            (p.get_x() + p.get_width() / 2., height), 
            ha='center', 
            va='center', 
            xytext=(0, 9), 
            textcoords='offset points',
            fontsize=9, # Reduced font size a bit
            color='black'
        )

    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300)
    print(f"\nSummary plot saved to: {save_path}")
    plt.close()

def analyze_experiment_results(results_dir):
    """
    Scans a results directory for all 'segmentation_sweep_results.csv' files,
    parses them, and prints a master summary table.
    """
    
    # 1. Find all the segmentation sweep CSVs
    search_path = os.path.join(results_dir, "**/segmentation_sweep_results.csv")
    csv_files = glob.glob(search_path, recursive=True)
    
    if not csv_files:
        print(f"Error: No 'segmentation_sweep_results.csv' files found under {results_dir}")
        print("Please run the main experiment first.")
        return

    print(f"Found {len(csv_files)} result files. Parsing...")
    
    all_dfs = []
    for f in csv_files:
        try:
            # Extract class name from the directory path
            class_name = os.path.basename(os.path.dirname(f))
            df = pd.read_csv(f)
            df['class_name'] = class_name
            all_dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not parse {f}. Error: {e}")
            
    if not all_dfs:
        print("Error: No dataframes were loaded.")
        return

    master_df = pd.concat(all_dfs)

    # 2. Filter data
    # We only care about IoU on anomalous images
    anomalous_df = master_df[master_df['gt_label'] == 1].copy()
    
    # 3. Create the summary
    summary_data = []
    
    all_classes = anomalous_df['class_name'].unique()
    
    for class_name in sorted(all_classes):
        class_df = anomalous_df[anomalous_df['class_name'] == class_name]
        
        # A. Find the best "NormalPercentile" (our new method)
        np_df = class_df[class_df['strategy'] == 'NormalPercentile']
        if not np_df.empty:
            best_np_row = np_df.loc[np_df['iou'].idxmax()]
            best_np_param = best_np_row['param']
            best_np_iou = best_np_row['iou']
            best_np_dice = best_np_row['dice']
        else:
            best_np_param = "N/A"
            best_np_iou = 0.0
            best_np_dice = 0.0
            
        # B. Find the "ClassF1" (original) method's score
        f1_df = class_df[class_df['strategy'] == 'ClassF1']
        if not f1_df.empty:
            # All ClassF1 rows for a class have the same IoU
            f1_iou = f1_df['iou'].iloc[0] 
            f1_dice = f1_df['dice'].iloc[0]
        else:
            f1_iou = 0.0
            f1_dice = 0.0
            
        summary_data.append({
            "Class": class_name,
            "Best_NP_Param (%)": best_np_param, # Store as number
            "Best_NP_IoU": best_np_iou,
            "ClassF1_IoU": f1_iou,
            "Best_NP_Dice": best_np_dice,
            "ClassF1_Dice": f1_dice
        })

    if not summary_data:
        print("Error: No summary data could be generated.")
        return

    summary_df = pd.DataFrame(summary_data)
    summary_df.set_index("Class", inplace=True)
    
    # Convert param column to numeric *before* calculating stats
    summary_df["Best_NP_Param (%)"] = pd.to_numeric(summary_df["Best_NP_Param (%)"])
    
    # --- NEW: Add STD_DEV Row ---
    # Calculate stats, dropping any non-numeric rows just in case
    numeric_df = summary_df.select_dtypes(include=np.number)
    summary_df.loc['AVERAGE'] = numeric_df.mean()
    summary_df.loc['STD_DEV'] = numeric_df.std()
    # --- END NEW ---

    # 4. Create the summary plot
    # We do this *before* formatting the columns as strings
    # We drop the stats rows so they don't get plotted as classes
    plot_save_path = os.path.join(results_dir, "master_segmentation_summary.png")
    df_for_plot = summary_df.drop(['AVERAGE', 'STD_DEV'], errors='ignore')
    plot_summary_chart(df_for_plot.copy(), plot_save_path) # Pass a copy to the plotter

    # 5. Save the summary table to a CSV
    # Save the *full* dataframe with numeric stats
    save_path = os.path.join(results_dir, "master_segmentation_summary.csv")
    summary_df.to_csv(save_path)
    print(f"\nSummary table saved to: {save_path}")

    # 6. Print the human-readable table
    print("\n--- MASTER SEGMENTATION SUMMARY ---")
    print(f"Comparing 'ClassF1' (Original) vs. Best 'NormalPercentile' (New)\n")
    
    # Format for clean printing
    summary_df['Best_NP_Param (%)'] = summary_df['Best_NP_Param (%)'].map('{:.1f}'.format).replace('nan', '---')
    summary_df['Best_NP_IoU'] = summary_df['Best_NP_IoU'].map('{:.4f}'.format)
    summary_df['ClassF1_IoU'] = summary_df['ClassF1_IoU'].map('{:.4f}'.format)
    summary_df['Best_NP_Dice'] = summary_df['Best_NP_Dice'].map('{:.4f}'.format)
    summary_df['ClassF1_Dice'] = summary_df['ClassF1_Dice'].map('{:.4f}'.format)

    print(summary_df.to_string())
    
    # 7. Find the final recommended parameter
    # We must drop the stats rows before calculating the median
    final_param_recommendation = summary_df.drop(['AVERAGE', 'STD_DEV'], errors='ignore')['Best_NP_Param (%)'].astype(float).median()
    print("\n--- FINAL RECOMMENDATION ---")
    print(f"The MEDIAN of the 'Best_NP_Param' across all classes is: {final_param_recommendation:.1f}%")
    print(f"This is your single, most robust thresholding parameter to use for your Bowtie dataset.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaDiM Sweep Results Analyzer")
    parser.add_argument(
        "results_dir",
        type=str,
        help="The root results directory containing the class subfolders (e.g., './results/mvtec_test_sweep')",
    )
    args = parser.parse_args()
    
    analyze_experiment_results(args.results_dir)