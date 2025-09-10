import pandas as pd
import matplotlib.pyplot as plt
import io


data_text = """
lr 1e-6
Accuracy EXACT: 0.43333333333333335
Accuracy APPROX: 0.6833333333333333
lr 1e-5
Accuracy EXACT: 0.43333333333333335
Accuracy APPROX: 0.6833333333333333
lr 5e-5
Accuracy EXACT: 0.45
Accuracy APPROX: 0.6833333333333333
lr 8e-5
Accuracy EXACT: 0.45
Accuracy APPROX: 0.6833333333333333
lr 1e-4
Accuracy EXACT: 0.45
Accuracy APPROX: 0.6833333333333333
lr 2e-4
Accuracy EXACT: 0.43333333333333335
Accuracy APPROX: 0.65
lr 5e-4
Accuracy EXACT: 0.48333333333333334
Accuracy APPROX: 0.7
lr 8e-4
Accuracy EXACT: 0.48333333333333334
Accuracy APPROX: 0.6833333333333333
lr 1e-3
Accuracy EXACT: 0.36666666666666664
Accuracy APPROX: 0.5833333333333334
lr 5e-3
Accuracy EXACT: 0.43333333333333335
Accuracy APPROX: 0.6
lr 1e-2
Accuracy EXACT: 0.3
Accuracy APPROX: 0.4166666666666667
"""

# Correctly parse the data from the text
data = []
lines = data_text.strip().split('\n')
for i in range(0, len(lines), 3):
    lr_line = lines[i]
    exact_line = lines[i+1]
    approx_line = lines[i+2]

    lr = float(lr_line.split()[-1])
    exact_accuracy = float(exact_line.split()[-1])
    approx_accuracy = float(approx_line.split()[-1])

    data.append([lr, exact_accuracy, approx_accuracy])

df = pd.DataFrame(data, columns=['learning_rate', 'exact_accuracy', 'approx_accuracy'])
df = df.sort_values(by='learning_rate')

baseline_exact_accuracy = 0.4166666666666667
baseline_approx_accuracy = 0.6333333333333333

plt.figure(figsize=(12, 6))
plt.plot(df['learning_rate'], df['exact_accuracy'], marker='o', linestyle='-', label='Exact Accuracy', linewidth=5)
plt.plot(df['learning_rate'], df['approx_accuracy'], marker='o', linestyle='-', label='Approximate Accuracy', linewidth=5)

plt.axhline(y=baseline_exact_accuracy, color='r', linestyle='--', label='Baseline Exact Accuracy', linewidth=5)
plt.axhline(y=baseline_approx_accuracy, color='g', linestyle='--', label='Baseline Approximate Accuracy', linewidth=5)

plt.xscale('log')
plt.xlabel('Learning Rate (log scale)', fontsize=24, fontweight='bold')
plt.ylabel('Accuracy', fontsize=24, fontweight='bold')
plt.title('Effect of Learning Rate on Model Performance', fontsize=24, fontweight='bold', pad=20)
#increase tick size and make them bold
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig('lr_ablation_study_with_baseline.png')