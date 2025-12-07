import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

# 1. Load and Filter Data
df = pd.read_csv('results/test_blob_256//color_profile_1/color_profile_1_blob_metrics.csv')
df_995 = df[df['Percentile'] == 99.5].copy()

# 2. Define Groups
df_995['Actual_Status'] = df_995['Class_Tag'].apply(lambda x: 'Defective' if x in ['TP', 'FN'] else 'Good')

# 3. Define Features to Analyze against Location (Centroid_Dist)
features_to_analyze = ['Area_Pixels', 'Circularity', 'Mean_Intensity', 'Peak_Intensity', 'Polar_Angle']
location_col = 'Centroid_Dist'

# --- PART 1: NUMERICAL ANALYSIS ---

# A. Correlation with Centroid_Dist (2 Classes and 4 Classes)
corr_data = []

# Function to get correlation
def get_corr(data, group_name, group_val):
    if len(data) < 3: return None
    res = {'Group_Type': group_name, 'Group_Value': group_val}
    for feat in features_to_analyze:
        c, p = spearmanr(data[feat], data[location_col])
        res[f'{feat}_Corr'] = c
        res[f'{feat}_Pval'] = p
    return res

# 2-Class Split
for status in ['Defective', 'Good']:
    sub = df_995[df_995['Actual_Status'] == status]
    corr_data.append(get_corr(sub, 'Actual_Status', status))

# 4-Class Split
for tag in ['TP', 'FN', 'TN', 'FP']:
    sub = df_995[df_995['Class_Tag'] == tag]
    corr_data.append(get_corr(sub, 'Class_Tag', tag))

corr_df = pd.DataFrame(corr_data)

# B. Statistical Difference Tests (Mann-Whitney U)
# Defective vs Good
mw_data = []
for feat in features_to_analyze + [location_col]:
    g1 = df_995[df_995['Actual_Status'] == 'Defective'][feat]
    g2 = df_995[df_995['Actual_Status'] == 'Good'][feat]
    if len(g1)>0 and len(g2)>0:
        stat, p = mannwhitneyu(g1, g2)
        mw_data.append({'Feature': feat, 'Comparison': 'Defective vs Good', 'P_Value': p, 'Significant': p<0.05})

# Misclassification Analysis (FP vs TN, FN vs TP)
for feat in features_to_analyze + [location_col]:
    # FP vs TN
    fp = df_995[df_995['Class_Tag'] == 'FP'][feat]
    tn = df_995[df_995['Class_Tag'] == 'TN'][feat]
    if len(fp)>0 and len(tn)>0:
        stat, p = mannwhitneyu(fp, tn)
        mw_data.append({'Feature': feat, 'Comparison': 'FP vs TN (Noise Check)', 'P_Value': p, 'Significant': p<0.05})
    
    # FN vs TP
    fn = df_995[df_995['Class_Tag'] == 'FN'][feat]
    tp = df_995[df_995['Class_Tag'] == 'TP'][feat]
    if len(fn)>0 and len(tp)>0:
        stat, p = mannwhitneyu(fn, tp)
        mw_data.append({'Feature': feat, 'Comparison': 'FN vs TP (Defect Check)', 'P_Value': p, 'Significant': p<0.05})

mw_df = pd.DataFrame(mw_data)

# Save tables
corr_df.to_csv('comprehensive_correlations_995.csv', index=False)
mw_df.to_csv('comprehensive_stats_tests_995.csv', index=False)

print("Correlation Analysis:")
print(corr_df.to_markdown(index=False, numalign="left", stralign="left"))
print("\nStatistical Difference Tests:")
print(mw_df.to_markdown(index=False, numalign="left", stralign="left"))

# --- PART 2: VISUALIZATION ---

# Set style
sns.set_style("whitegrid")
colors_2class = {'Defective': 'red', 'Good': 'green'}
colors_4class = {'TP': 'red', 'FN': 'orange', 'TN': 'green', 'FP': 'blue'}
markers_4class = {'TP': 'o', 'FN': 'X', 'TN': 'o', 'FP': 'D'}

# 1. Location Distribution (Polar Plot) - Individual
fig = plt.figure(figsize=(10, 5))
# 2 Class
ax1 = fig.add_subplot(121, projection='polar')
for status, color in colors_2class.items():
    sub = df_995[df_995['Actual_Status'] == status]
    # Convert degrees to radians
    theta = np.deg2rad(sub['Polar_Angle'])
    r = sub['Centroid_Dist']
    ax1.scatter(theta, r, c=color, label=status, alpha=0.6)
ax1.set_title("Location Map (Defective vs Good)")
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize='small')

# 4 Class
ax2 = fig.add_subplot(122, projection='polar')
for tag, color in colors_4class.items():
    sub = df_995[df_995['Class_Tag'] == tag]
    theta = np.deg2rad(sub['Polar_Angle'])
    r = sub['Centroid_Dist']
    ax2.scatter(theta, r, c=color, label=tag, marker=markers_4class[tag], alpha=0.6)
ax2.set_title("Location Map (4 Classes)")
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize='small')
plt.tight_layout()
plt.savefig('comp_polar_map_995.png')


# Loop for Scatter Plots: Feature vs Distance
for feat in ['Mean_Intensity', 'Peak_Intensity', 'Area_Pixels', 'Circularity']:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 2 Class Plot
    sns.scatterplot(data=df_995, x='Centroid_Dist', y=feat, hue='Actual_Status', 
                    palette=colors_2class, ax=axes[0], alpha=0.7)
    axes[0].set_title(f'{feat} vs Distance (Defective vs Good)')
    
    # 4 Class Plot
    sns.scatterplot(data=df_995, x='Centroid_Dist', y=feat, hue='Class_Tag', 
                    palette=colors_4class, style='Class_Tag', markers=markers_4class, ax=axes[1], alpha=0.7)
    axes[1].set_title(f'{feat} vs Distance (4 Classes)')
    
    if feat == 'Area_Pixels':
        axes[0].set_yscale('log')
        axes[1].set_yscale('log')
        
    plt.tight_layout()
    plt.savefig(f'comp_{feat}_vs_dist_995.png')

# Boxplots for direct comparison of distributions
for feat in ['Mean_Intensity', 'Peak_Intensity', 'Area_Pixels', 'Circularity', 'Centroid_Dist']:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.boxplot(data=df_995, x='Actual_Status', y=feat, palette=colors_2class, ax=axes[0])
    axes[0].set_title(f'{feat} Distribution (Defective vs Good)')
    
    sns.boxplot(data=df_995, x='Class_Tag', y=feat, palette=colors_4class, order=['TP', 'FN', 'FP', 'TN'], ax=axes[1])
    axes[1].set_title(f'{feat} Distribution (4 Classes)')
    
    if feat == 'Area_Pixels':
        axes[0].set_yscale('log')
        axes[1].set_yscale('log')
        
    plt.tight_layout()
    plt.savefig(f'comp_boxplot_{feat}_995.png')