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

data = """
Model,Size,Type,Exact Match,Approx Match
SmolLM2-1.7B,1700000000,baseline,0.1162,0.1320
SmolLM2-1.7B,1700000000,func,0.1866,0.1972
Qwen3-8B,8000000000,baseline,0.7342,0.7887
Qwen3-8B,8000000000,func,0.7201,0.7359
Llama3.2-1B,1000000000,baseline,0.0141, 0.0141
Llama3.2-1B,1000000000,func,0.0440,0.0546
Llama3.2-3B,3000000000,baseline,0.1197,0.1285
Llama3.2-3B,3000000000,func,0.1461,0.1708
Meta-Llama3-8B,8000000000,baseline,0.2553,0.3063
Meta-Llama3-8B,8000000000,func,0.3697,0.3996
Phi-4-14B,14000000000,baseline,0.7254,0.8116
Phi-4-14B,14000000000,func,0.6972,0.7606
Gemma3-4B,4000000000,baseline,0.1919,0.2095
Gemma3-4B,4000000000,func,0.2764,0.2993
Gemma3-12B,12000000000,func,0.6077,0.6254
Gemma3-12B,12000000000,baseline,0.5317,0.5915
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
bars = ax.bar(df_gain_sorted.index, df_gain_sorted['Relative Gain (%)'], color=colors, width=0.4)

ax.set_ylabel('Relative Improvement (%)', fontsize=16, fontweight='bold')
ax.set_title('Relative Performance Gain of Toolken over Baseline (Exact)', fontsize=18, fontweight='bold')
plt.xticks(rotation=45, ha="right", fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
ax.axhline(0, color='black', linewidth=0.8) # Zero line for reference

# Add text labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}%', va='bottom' if yval >= 0 else 'top', ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('relative_gain_plot.png')
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
fig, ax = plt.subplots(figsize=(15, 8))

# Draw lines connecting baseline and func points for each model
for i, row in df_merged.iterrows():
    #increase the line width to 2
    ax.plot([row['Size_base'], row['Size_func']],
            [row['Exact Match_base'], row['Exact Match_func']],
            color='gray', linestyle='-', marker='', zorder=1, linewidth=3)

# Plot the points for baseline and func
ax.scatter(df_merged['Size_base'], df_merged['Exact Match_base'], color='dodgerblue', s=200, label='Baseline', zorder=2)
ax.scatter(df_merged['Size_func'], df_merged['Exact Match_func'], color='darkorange', s=200, label='Toolken', zorder=2)

# Set labels and title
ax.set_xlabel('Model Size (Parameters)', fontsize=24, fontweight='bold')
ax.set_ylabel('Exact Match (%)', fontsize=24, fontweight='bold')
#give small pad between title and plot
ax.set_title('Performance vs. Model Size', fontsize=26, fontweight='bold', pad=20)
#set x ticks to be more bigger and bold
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

#make legend font size bigger and bold
legend_properties = {'weight':'bold', 'size': 20}
ax.legend(fontsize=30, prop=legend_properties)
#change legend to Baseline and Toolken

#for x axis convert them to billions with B suffix, also make them BOLD
def billions(x, pos):
    'The two args are the value and tick position'
    return f'{x*1e-9:.1f}B'.replace('.0B', 'B')
ax.xaxis.set_major_formatter(plt.FuncFormatter(billions))
ax.grid(axis='y', linestyle='--', alpha=0.7)
#make x ticks bold
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
#make y ticks bold
for label in ax.get_yticklabels():
    label.set_fontweight('bold')


# Add text labels for each model
for i, row in df_merged.iterrows():

    #check that 14B is in the model name

    #discard the -X B in the model name for better visualization
    if '8B' in row['Model_Sized']:
        model_name = row['Model_Sized'].replace('-8B', '')
    if '4B' in row['Model_Sized']:
        model_name = row['Model_Sized'].replace('-4B', '')
    if '1B' in row['Model_Sized']:
        model_name = row['Model_Sized'].replace('-1B', '')
    if '3B' in row['Model_Sized']:
        model_name = row['Model_Sized'].replace('-3B', '')
    if '1.7B' in row['Model_Sized']:
        model_name = row['Model_Sized'].replace('-1.7B', '')
    if '12B' in row['Model_Sized']:
        model_name = row['Model_Sized'].replace('-12B', '')
    if '14B' in row['Model_Sized']:
        print("Found 14B model")
        model_name = row['Model_Sized'].replace('-14B', '')
 
    print(row['Model_Sized'])
    #align to left of the func point
    if row['Model_Sized'] == 'Qwen3-8B-8B':
        ax.text(row['Size_func'] * 1.03 , row['Exact Match_func'] * 1.04, model_name, fontsize=20, fontweight='bold', color='black', ha='right')
    elif row['Model_Sized'] == 'Llama-3-8B-8B':
        ax.text(row['Size_func'] * 1.03 , row['Exact Match_func'] * 1.05, model_name, fontsize=20, fontweight='bold', color='black', ha='right')
    elif row['Model_Sized'] == 'Gemma-3-4B-4B':
        ax.text(row['Size_func'] * 1.1 , row['Exact Match_func'] * 1.01, model_name, fontsize=20, fontweight='bold', color='black', ha='right')
    elif row['Model_Sized'] == 'Gemma3-12B-12B':
        ax.text(row['Size_func'] * 1.05 , row['Exact Match_func'] * 1.03, model_name, fontsize=20, fontweight='bold', color='black', ha='right')
    elif row['Model_Sized'] == 'SmolLM2-1.7B-1.7B':
        ax.text(row['Size_func'] * 1.4 , row['Exact Match_func'] * 1.02, model_name, fontsize=20, fontweight='bold', color='black', ha='right')
    elif row['Model_Sized'] == 'Llama-3.2-3B-3B':
        ax.text(row['Size_func'] * 0.9 , row['Exact Match_func'] * 1.05, model_name, fontsize=20, fontweight='bold', color='black', ha='left')
    elif row['Model_Sized'] == 'Phi-4-14B-14B':
        ax.text(row['Size_func'] * 0.95 , row['Exact Match_func'] * 1.05, model_name, fontsize=20, fontweight='bold', color='black', ha='left')

    else:
        ax.text(row['Size_func'] * 0.9 , row['Exact Match_func'] * 1.1, model_name, fontsize=20, fontweight='bold', color='black',  ha='left')

plt.tight_layout()
plt.savefig('performance_vs_size_plot.png')
print("Generated performance_vs_size_plot.png")

# Display the plots if in an interactive environment
# plt.show()
