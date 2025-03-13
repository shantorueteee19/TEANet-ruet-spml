# Plotting confusion matrix
def normalized_conf_matrix(conf):
  class_names = ['Normal', 'Stressed']
  row_sums = conf.sum(axis=1, keepdims=True)
  normalized_conf = conf / row_sums
  plt.figure(figsize=(4, 3))
  sns.heatmap(normalized_conf, annot=True, fmt=".3f", cmap='Blues', cbar=True, xticklabels=class_names, yticklabels=class_names)
  plt.title('Normalized Confusion Matrix')
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.show()
final_conf_matrix = np.mean(conf_matrices, axis=0)
normalized_conf_matrix(final_conf_matrix)

# Plot ROC curve and AUC
fpr, tpr, _ = roc_curve(all_y_true, all_y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(4, 3))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve')
plt.plot([0, 1], [0, 1], color='green', lw=2, linestyle='-.')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=9)
plt.ylabel('True Positive Rate', fontsize=9)
plt.title('Receiver Operating Characteristic', fontsize=10)
plt.legend(loc="lower right")
plt.show()

# Plotting of accuracy and validation accuracy
for history,i in zip(histories,sub):
    print(f"In sub - {i}")
    plt.figure(figsize=(5, 3))
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='darkorange')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='blue')

    # Calculate standard deviation
    train_acc_std = np.std(history.history['accuracy'])
    val_acc_std = np.std(history.history['val_accuracy'])

    # Add standard deviation to the plot
    plt.fill_between(range(len(history.history['accuracy'])),
                      np.array(history.history['accuracy']) - train_acc_std,
                      np.array(history.history['accuracy']) + train_acc_std,
                      color='darkorange', alpha=0.2)
    plt.fill_between(range(len(history.history['val_accuracy'])),
                      np.array(history.history['val_accuracy']) - val_acc_std,
                      np.array(history.history['val_accuracy']) + val_acc_std,
                      color='blue', alpha=0.2)

    plt.title('Accuracy vs Epochs', fontsize=10)
    plt.xlabel('Epochs', fontsize=9)
    plt.ylabel('Accuracy', fontsize=9)
    plt.legend(labels=['Training', 'Validation'], fontsize=7, loc='upper left')
    plt.show()

# Plotting of loss and validation loss
for history,i in zip(histories,sub):
    print(f"In sub - {i}")
    plt.figure(figsize=(5, 3))
    plt.plot(history.history['loss'], label='Training Loss', color='darkorange')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='blue')

    # Calculate standard deviation
    train_loss_std = np.std(history.history['loss'])
    val_loss_std = np.std(history.history['val_loss'])

    # Add standard deviation to the plot
    plt.fill_between(range(len(history.history['loss'])),
                      np.array(history.history['loss']) - train_loss_std,
                      np.array(history.history['loss']) + train_loss_std,
                      color='darkorange', alpha=0.2)
    plt.fill_between(range(len(history.history['val_loss'])),
                      np.array(history.history['val_loss']) - val_loss_std,
                      np.array(history.history['val_loss']) + val_loss_std,
                      color='blue', alpha=0.2)

    plt.title('Loss vs Epochs', fontsize=10)
    plt.xlabel('Epochs', fontsize=9)
    plt.ylabel('Loss', fontsize=9)
    plt.legend(labels=['Training', 'Validation'], fontsize=7, loc='upper left')
    plt.show()
