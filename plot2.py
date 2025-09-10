import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
import re

# Raw data provided by the user in a multi-line string format
raw_data = """
MODEL Llama-3.2-1B
FUNC
Accuracy EXACT: 0.13333333333333333
Accuracy APPROX: 0.21666666666666667
BASELINE
Accuracy EXACT: 0.06666666666666667
Accuracy APPROX: 0.1
-----
MODEL Llama-3.2-3B
FUNC
Accuracy EXACT: 0.23333333333333334
Accuracy APPROX: 0.3333333333333333
BASELINE
Accuracy EXACT: 0.08333333333333333
Accuracy APPROX: 0.11666666666666667
----
MODEL SmolLM2-1.7B
FUNC
Accuracy EXACT: 0.31666666666666665
Accuracy APPROX: 0.4
BASELINE
Accuracy EXACT: 0.11666666666666667
Accuracy APPROX: 0.23333333333333334
----
MODEL Gemma-3-4B
FUNC
Accuracy EXACT: 0.4166666666666667
Accuracy APPROX: 0.6
BASELINE
Accuracy EXACT: 0.18333333333333332
Accuracy APPROX:  0.23333333333333334
----
MODEL Llama-3-8B
FUNC
Accuracy EXACT: 0.16666666666666666
Accuracy APPROX: 0.25
BASELINE
Accuracy EXACT: 0.15
Accuracy APPROX: 0.2
----
MODEL Qwen3-8B
FUNC
Accuracy EXACT: 0.36666666666666664
Accuracy APPROX: 0.5833333333333334
BASELINE
Accuracy EXACT: 0.4166666666666667
Accuracy APPROX: 0.6333333333333333
"""



# --- Data Parsing and Preparation ---

def parse_size_from_name(name):
    """Extracts numeric size from model name string (e.g., '1.7B' -> 1.7e9)."""
    # Find patterns like 8B, 4b, 1.7B, case-insensitive
    match = re.search(r'(\d+(\.\d+)?)[bB]', name)
    if match:
        size_val = float(match.group(1))
        return int(size_val * 1e9)
    return 0 # Default size if not found

# Process the raw data string block by block
records = []
# FIX: Use a regular expression to split on four OR MORE dashes for robustness
blocks = re.split(r'-{4,}', raw_data.strip())

for block in blocks:
    if not block.strip():
        continue
    
    lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
    
    model_name_full = lines[0].replace('MODEL ', '')
    model_size = parse_size_from_name(model_name_full)
    # FIX: Simplify model name by removing the size suffix, preserving the base name
    model_name_simple = re.sub(r'-\d+(\.\d+)?[bB]', '', model_name_full)

    # Func data
    func_exact = float(lines[2].split(': ')[1]) * 100
    func_approx = float(lines[3].split(': ')[1]) * 100
    records.append([model_name_simple, 'Toolken', func_exact, func_approx, model_size])
    
    # Baseline data
    baseline_exact = float(lines[5].split(': ')[1]) * 100
    baseline_approx = float(lines[6].split(': ')[1]) * 100
    records.append([model_name_simple, 'Baseline', baseline_exact, baseline_approx, model_size])
    
# Create the pandas DataFrame
df = pd.DataFrame(records, columns=['Model', 'Type', 'Exact Match', 'Approx Match', 'Size'])

#order by size
df = df.sort_values(by=['Size', 'Model', 'Type'], ascending=[True, True, True]).reset_index(drop=True)
print(df)

# --- Code from user's example, with minor adaptations ---

# Function to format the size for labels
def format_size(s):
    if s >= 1e9:
        return f'{s/1e9:.1f}B'.replace('.0B', 'B')
    return str(s)

# Create a unique model name with size for sorting and labeling
df['Model_Sized'] = df.apply(lambda row: f"{row['Model']}-{format_size(row['Size'])}", axis=1)
# Calculate the 'Partial Match' score for the stacked bar
df['Partial Match'] = df['Approx Match'] - df['Exact Match']

# Ensure 'Baseline' always comes before 'Toolken' by setting Type as a categorical
df['Type'] = pd.Categorical(df['Type'], categories=['Baseline', 'Toolken'], ordered=True)

# Sort data for grouped plotting logic
df_sorted = df.sort_values(by=['Size', 'Model','Type'])

#sort by size, m

# --- Bar Position Calculation ---
bar_width = 0.8
group_gap = 0.2  # Gap between Baseline and Func
model_gap = 1.0   # Gap between different model groups

positions = []
tick_positions = []
tick_labels = []
current_pos = 0
last_model_sized = None

# Calculate the precise x-coordinate for each bar
for i, row in df_sorted.iterrows():
    if last_model_sized and row['Model_Sized'] != last_model_sized:
        # New model group, add the larger gap
        current_pos += model_gap
        # Calculate the center of the previous group for the tick label
        tick_positions.append((positions[-1] + positions[-2]) / 2)
        tick_labels.append(last_model_sized)
    elif last_model_sized:
        # Same model group, add the smaller gap
        current_pos += group_gap

    positions.append(current_pos)
    current_pos += bar_width
    last_model_sized = row['Model_Sized']

# Add the last group's tick position and label
tick_positions.append((positions[-1] + positions[-2]) / 2)
tick_labels.append(last_model_sized)


# --- Plotting ---
fig, ax = plt.subplots(figsize=(16, 8))

# Plot the stacked bars (Exact Match first, then Partial Match on top)
ax.bar(positions, df_sorted['Exact Match'], width=bar_width, label='Exact Match', color='dodgerblue')
ax.bar(positions, df_sorted['Partial Match'], width=bar_width, bottom=df_sorted['Exact Match'], label='Partial Match', color='mediumseagreen')

# Add annotations (Baseline/Func) below the bars for clarity
y_offset_type = -5  # How far below the x-axis to place the type label
for pos, type_label in zip(positions, df_sorted['Type']):
    ax.text(pos, y_offset_type, type_label, ha='center', va='top', fontsize=10, color='black', fontweight='bold')

# Add total value labels on top of each bar
for pos, total in zip(positions, df_sorted['Approx Match']):
    ax.text(pos, total + 0.5, f'{total:.2f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

# --- Final Touches ---
ax.set_ylabel('Performance Score (%)', fontsize=20, fontweight='bold')
ax.set_title('Grouped Performance Breakdown: Baseline vs. Toolken', fontsize=20, fontweight='bold')
ax.legend(fontsize=20)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Set the x-ticks and labels to be centered under each model group, increase tick size and make them bold
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=16, fontweight='bold')
#put yticks from 0 to 100 with step 10, increase tick size and make them bold
ax.set_yticks(np.arange(0, 81, 10))
#make them bold

ax.tick_params(axis='y', labelsize=24)

# Adjust y-axis limits to make space for annotations below
ax.set_ylim(bottom=y_offset_type - 10)

plt.tight_layout()
# Adjust layout to prevent labels from being cut off
plt.subplots_adjust(bottom=0.2)
plt.savefig('grouped_stacked_performance_funcqa.png')
print("Generated plot and saved as 'grouped_stacked_performance_funcqa.png'")
# plt.show() # Uncomment to display the plot directly if running locally

