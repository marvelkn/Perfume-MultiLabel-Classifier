if __name__ == "__main__":
    raise SystemExit("Historical script retired. Use the versioned pipeline in README.md; legacy outputs are not current evidence.")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
df = pd.read_csv('reports/comparison_table.csv')

# Melt dataframe for seaborn
df_melt = df.melt(id_vars='Metrik', var_name='Model', value_name='Skor')

# Plot setup
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Create barplot
ax = sns.barplot(x='Metrik', y='Skor', hue='Model', data=df_melt, palette=['#1f77b4', '#ff7f0e'])

# Add title and labels
plt.title('Perbandingan Performa XGBoost vs LightGBM', fontsize=16, pad=15)
plt.ylabel('Skor', fontsize=12)
plt.xlabel('Metrik Evaluasi', fontsize=12)
plt.ylim(0, 1.0)
plt.legend(title='Model')

# Add values on top of bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.3f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points',
                   fontsize=10)

plt.tight_layout()

# Save plot
out_dir = r"C:\Users\Lenovo\Documents\UMN\Semester 7\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael\assets\pics"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, 'hasil_evaluasi.png'), dpi=300)
print(f"Plot saved to {os.path.join(out_dir, 'hasil_evaluasi.png')}")
