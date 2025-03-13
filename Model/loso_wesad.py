import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, auc, recall_score, classification_report, precision_score, roc_auc_score, cohen_kappa_score
from keras.callbacks import EarlyStopping
from tensorflow.keras import models
# Initialize lists to store metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
auc_scores = []
kappa_scores = []
conf_matrices = []
histories = []
# Lists to store true labels and predicted probabilities for ROC AUC calculation
all_y_true = []
all_y_pred_prob = []
#RUET SPML 19 subjects
sub = [f'{i}' for i in range(1,20)] #add sub_file names
# Loop over each fold
for test_sub in sub:
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    #Load train files
    for subL in sub:
        if subL == test_sub:
            continue
        else: 
            #Normal
            dfNormal = pd.read_csv(fr'/dir/normal{subL}.csv', header = None)
            df_train = pd.concat([df_train,dfNormal], axis =0, ignore_index = True)
            #Stress
            dfStress = pd.read_csv(fr'/dir/stress{subL}.csv', header = None)
            #minority class augmentation
            dfStress = minorityClass_aug(dfStress, window_len=30, samp_freq=64, n_majority=dfNormal.shape[0], label = 'stressed')
            df_train = pd.concat([df_train,dfStress], axis =0, ignore_index = True)
    #Test data
    dfNormal = pd.read_csv(fr'/dir/normal{test_sub}.csv', header = None)
    df_test = pd.concat([df_test,dfNormal], axis =0, ignore_index = True)
    #Stress
    dfStress = pd.read_csv(fr'/dir/stress{test_sub}.csv', header = None)        
    df_test = pd.concat([df_test,dfStress], axis =0, ignore_index = True)
    #Separation of signal windows and 
    X_train = df_train.iloc[:, 0:-1].values
    y_train = df_train.iloc[:, -1]
    X_test = df_test.iloc[:, 0:-1].values
    y_test = df_test.iloc[:, -1]
    #Map labels
    y_map = {
        'normal':0,
        'stressed': 1,
    }
    y_train = y_train.map(y_map)
    y_test = y_test.map(y_map)
    # Standardize the signal windows
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    input_shape = (X_train.shape[1], 1)
    #Call model
    num_classes = df_train.iloc[:, -1].nunique()
    model = TEANet_model(input_shape, num_classes)
    # Compile the model
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    #Early Stopping
    es = EarlyStopping(monitor='val_accuracy', patience=70, verbose=1, mode='max', restore_best_weights=True)
    print(f"_____Test Sub: {test_sub}_____")
    # Train the model
    history = model.fit(X_train, y_train, epochs=1000, batch_size=64, validation_data=(X_test,y_test), callbacks=[es])
    # Evaluate the model on the test set
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    #Evaluation metrices calculation
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall_in = recall_score(y_test, y_pred)
    precision_score_in = precision_score(y_test, y_pred)
    kappa_scores_in = cohen_kappa_score(y_test, y_pred)
    auc_scores_in = roc_auc_score(y_test, y_pred_prob[:, 1])
    #Append to list
    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    conf_matrices.append(confusion_matrix(y_test, y_pred))
    histories.append(history)
    precision_scores.append(precision_score_in)
    recall_scores.append(recall_in)
    kappa_scores.append(kappa_scores_in)
    auc_scores.append(auc_scores_in)
    # Store true labels and predicted probabilities
    all_y_true.extend(y_test)
    all_y_pred_prob.extend(y_pred_prob[:, 1])
