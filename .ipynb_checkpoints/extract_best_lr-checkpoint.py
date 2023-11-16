import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from glob import glob

os.chdir('/home/kdoherty/spurge/data_release')

# Load all CSV files
path = './results/lr_tune'
all_files = glob(os.path.join(path, "*.csv"))
df_list = []

all_files

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    # Convert 'epoch_acc' to numeric, forcing non-numeric values to NaN
    df['epoch_acc'] = pd.to_numeric(df['epoch_acc'], errors='coerce')
    # Ensure 'learning_rate' and 'set' are treated as strings
    df['learning_rate'] = df['learning_rate'].astype(str)
    df['set'] = df['set'].astype(str)
    df_list.append(df)

# Concatenate all data into one DataFrame
results_df = pd.concat(df_list, axis=0, ignore_index=True)
grouped_df = results_df.groupby(['learning_rate', 'epoch'])['epoch_acc'].mean().reset_index()

# Plotting and saving the plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=grouped_df, x='epoch', y='epoch_acc', hue='learning_rate', palette='viridis')
plt.title('Cross-validated accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Cross-validated accuracy')
plt.legend(title='Learning Rate')

plt.savefig('./results/lr_find_plot.png')
plt.close()

# Finding the best epoch for each learning rate and saving the results
best_epochs_df = grouped_df.loc[grouped_df.groupby('learning_rate')['epoch_acc'].idxmax()]
best_epochs_df = best_epochs_df.rename(columns={'epoch_acc': 'accuracy'})
best_epochs_df = best_epochs_df[['learning_rate', 'epoch', 'accuracy']].reset_index(drop=True)

best_epochs_df.to_csv('./results/best_lr.csv', index=False)
