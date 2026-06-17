import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DRIVE_RESULTS_DIR = "results/train_20260513_171801_col_selector_rlcg"
# Read the CSV file
df = pd.read_csv(f'{DRIVE_RESULTS_DIR}/training_metrics.csv')

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Neural Network Training Metrics', fontsize=16, fontweight='bold')

# Plot 1: Loss over epochs
axes[0, 0].plot(df['epoch'], df['loss'], marker='o', linewidth=2, markersize=4, color='#e74c3c')
axes[0, 0].set_xlabel('Epoch', fontsize=11)
axes[0, 0].set_ylabel('Loss', fontsize=11)
axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Average Reward over epochs
axes[0, 1].plot(df['epoch'], df['avg_reward'], marker='s', linewidth=2, markersize=4, color='#27ae60')
axes[0, 1].set_xlabel('Epoch', fontsize=11)
axes[0, 1].set_ylabel('Average Reward', fontsize=11)
axes[0, 1].set_title('Average Reward', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Q-Value Mean over epochs
axes[1, 0].plot(df['epoch'], df['q_value_mean'], marker='^', linewidth=2, markersize=4, color='#3498db')
axes[1, 0].set_xlabel('Epoch', fontsize=11)
axes[1, 0].set_ylabel('Q-Value Mean', fontsize=11)
axes[1, 0].set_title('Q-Value Mean', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Epsilon over epochs
axes[1, 1].plot(df['epoch'], df['epsilon'], marker='d', linewidth=2, markersize=4, color='#9b59b6')
axes[1, 1].set_xlabel('Epoch', fontsize=11)
axes[1, 1].set_ylabel('Epsilon', fontsize=11)
axes[1, 1].set_title('Exploration Epsilon', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig(f'{DRIVE_RESULTS_DIR}/training_metrics.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to {DRIVE_RESULTS_DIR}/training_metrics.png")

# Display the plot
# plt.show()

# Print summary statistics
print("\n" + "="*50)
print("TRAINING METRICS SUMMARY")
print("="*50)
print(f"\nTotal Epochs: {len(df)}")
print(f"\nLoss:")
print(f"  Initial: {df['loss'].iloc[0]:.2f}")
print(f"  Final:   {df['loss'].iloc[-1]:.2f}")
print(f"  Min:     {df['loss'].min():.2f} (Epoch {df['loss'].idxmin() + 1})")
print(f"  Max:     {df['loss'].max():.2f} (Epoch {df['loss'].idxmax() + 1})")

print(f"\nAverage Reward:")
print(f"  Initial: {df['avg_reward'].iloc[0]:.2f}")
print(f"  Final:   {df['avg_reward'].iloc[-1]:.2f}")
print(f"  Min:     {df['avg_reward'].min():.2f} (Epoch {df['avg_reward'].idxmin() + 1})")
print(f"  Max:     {df['avg_reward'].max():.2f} (Epoch {df['avg_reward'].idxmax() + 1})")

print(f"\nQ-Value Mean:")
print(f"  Initial: {df['q_value_mean'].iloc[0]:.2f}")
print(f"  Final:   {df['q_value_mean'].iloc[-1]:.2f}")
print(f"  Min:     {df['q_value_mean'].min():.2f} (Epoch {df['q_value_mean'].idxmin() + 1})")
print(f"  Max:     {df['q_value_mean'].max():.2f} (Epoch {df['q_value_mean'].idxmax() + 1})")
