import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np

# Data provided by the user, formatted as a CSV string
data = """Model,Size,Type,Exact Match,Approx Match
Llama-3.2-1B,1000000000,func,0.13333333333333333,0.21666666666666667
Llama-3.2-1B,1000000000,baseline,0.06666666666666667,0.1
Llama-3.2-3B,3000000000,func,0.23333333333333334, 0.3333333333333333
Llama-3.2-3B,3000000000,baseline,0.08333333333333333, 0.11666666666666667
SmolLM2-1.7B,1700000000,func, 0.31666666666666665, 0.4
SmolLM2-1.7B,1700000000,baseline,0.11666666666666667,  0.23333333333333334
Gemma-3-4B,4000000000,func,0.4166666666666667,0.6
Gemma-3-4B,4000000000,baseline,0.18333333333333332,  0.23333333333333334
Llama-3-8B,8000000000,func,0.16666666666666666,0.25
Llama-3-8B,8000000000,baseline,0.15,0.2
Qwen3-8B,8000000000,func,0.36666666666666664,0.5833333333333334
Qwen3-8B,8000000000,baseline,0.4166666666666667,0.6333333333333333
"""

# Read the data into a pandas DataFrame
df = pd.read_csv(io.StringIO(data))
# Multiply accuracy scores by 100 to display as percentages
df['Exact Match'] = df['Exact Match'] * 100
df['Approx Match'] = df['Approx Match'] * 100


# --- Data Preparation ---
# Function to format the size for display
def format_size(s):
    if s >= 1e9:
        # Format as billions, removing '.0B' for whole numbers
        return f'{s/1e9:.1f}B'.replace('.0B', 'B')
    return str(s)

# Create a unique model name with its formatted size for labeling
df['Model_Sized'] = df.apply(lambda row: f"{row['Model'].split('_')[-1]}-{format_size(row['Size'])}", axis=1)

# --- Plot 1: Relative Performance Gain ---
# Pivot the table to have 'func' and 'baseline' as columns
df_pivot = df.pivot(index='Model_Sized', columns='Type', values='Exact Match')
# Calculate the percentage gain of 'func' over 'baseline'
df_pivot['Relative Gain (%)'] = ((df_pivot['func'] - df_pivot['baseline']) / df_pivot['baseline']) * 100
df_gain_sorted = df_pivot.sort_values(by='Relative Gain (%)', ascending=False)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
# Handle potential division by zero or empty baseline values
df_gain_sorted.replace([np.inf, -np.inf], np.nan, inplace=True)
df_gain_sorted.dropna(subset=['Relative Gain (%)'], inplace=True)
# Color bars green for positive gain, red for negative
colors = ['g' if x >= 0 else 'r' for x in df_gain_sorted['Relative Gain (%)']]
bars = ax.bar(df_gain_sorted.index, df_gain_sorted['Relative Gain (%)'], color=colors)

ax.set_ylabel('Relative Improvement (%)', fontsize=12, fontweight='bold')
ax.set_title('Relative Performance Gain of Func over Baseline (Exact Match)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha="right")
ax.axhline(0, color='black', linewidth=0.8) # Zero line for reference

# Add text labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}%', va='bottom' if yval >= 0 else 'top', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('relative_gain_plot_funcqa.png')
print("Generated relative_gain_plot.png")

# --- Plot 2: Performance vs. Model Size (Dumbbell Plot) ---
# Separate data for baseline and func types
df_baseline = df[df['Type'] == 'baseline'].rename(columns={'Exact Match': 'Exact Match_base', 'Size': 'Size_base'})
df_func = df[df['Type'] == 'func'].rename(columns={'Exact Match': 'Exact Match_func', 'Size': 'Size_func'})
# Merge them back together on the model name
df_merged = pd.merge(df_baseline[['Model_Sized', 'Size_base', 'Exact Match_base']],
                       df_func[['Model_Sized', 'Size_func', 'Exact Match_func']],
                       on='Model_Sized')
df_merged = df_merged.sort_values(by='Size_base')

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Draw lines connecting baseline and func points for each model
for i, row in df_merged.iterrows():
    ax.plot([row['Size_base'], row['Size_func']],
            [row['Exact Match_base'], row['Exact Match_func']],
            color='gray', linestyle='-', marker='', zorder=1)

# Plot the points for baseline and func
ax.scatter(df_merged['Size_base'], df_merged['Exact Match_base'], color='dodgerblue', s=80, label='Baseline', zorder=2)
ax.scatter(df_merged['Size_func'], df_merged['Exact Match_func'], color='darkorange', s=80, label='Func', zorder=2)

# Set labels and title
ax.set_xlabel('Model Size (Parameters)', fontsize=12, fontweight='bold')
ax.set_ylabel('Exact Match (%)', fontsize=12, fontweight='bold')
ax.set_title('Performance vs. Model Size: Baseline vs. Func', fontsize=14, fontweight='bold')
ax.legend()
ax.set_xscale('log') # Use a logarithmic scale for model size
ax.grid(True, which="both", ls="--", c='0.7')

# Format the x-axis to show human-readable sizes (e.g., 1B, 8B)
from matplotlib.ticker import FuncFormatter
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1e9:.1f}B'.replace('.0B','B') if x > 0 else 0))

# Add text labels for each model
for i, row in df_merged.iterrows():
    ax.text(row['Size_func'] * 1.1, row['Exact Match_func'], row['Model_Sized'], fontsize=8, va='center')

plt.tight_layout()
plt.savefig('performance_vs_size_plot_funcqa.png')
print("Generated performance_vs_size_plot_funcqa.png")

# Display the plots if in an interactive environment
# plt.show()
