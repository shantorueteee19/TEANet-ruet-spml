data = {
    'Subject': sub,
    'Accuracy': accuracy_scores,
    'Precision': precision_scores,
    'Recall': recall_scores,
    'F1-Score': f1_scores,
    'AUC': auc_scores,
    'Kappa': kappa_scores
}

df_results = pd.DataFrame(data)
# Calculate the mean for each column
avg_scores = df_results.mean()
# Add the average row to the DataFrame
df_results.loc['Average'] = avg_scores
print(df_results)