from imblearn.pipeline import make_pipeline as make_imb_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score , accuracy_score
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


task = ''
modal = ''

# Define PCA
PCA_number = 0.9

# CV
outer_cv = StratifiedKFold(n_splits=10, shuffle=True)
inner_cv = StratifiedKFold(n_splits=10, shuffle=True,random_state=None)

data = pd.read_csv()
features_task1_CV = data.iloc[:, 1:39]
features_task1_NLP = data.iloc[:, 39:60]
features_task2_CV = data.iloc[:, 60:98]
features_task2_NLP = data.iloc[:, 98:119]
features_task3_CV = data.iloc[:, 119:157]
features_task3_NLP = data.iloc[:, 157:178]
self_esteem = data.iloc[:, -1]
low_self_esteem = (self_esteem < 31).astype(int)
ids = data.iloc[:, 0]

# Initializes the list that stores all experiment results
overall_results = []

# Pipeline
pipeline_lda_1CV = make_imb_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis()
    )
pipeline_lda_1NLP = make_imb_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis()
    )

pipeline_lda_2CV = make_imb_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis()
    )

pipeline_lda_2NLP = make_imb_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis()
    )

pipeline_lda_3CV = make_imb_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis()
    )

pipeline_lda_3NLP = make_imb_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis()
    )

pipeline_svc_1CV = make_imb_pipeline(
        StandardScaler(),
        PCA(n_components=PCA_number),
        SVC(C=1, kernel ='rbf', random_state=42)
    )

pipeline_svc_1NLP = make_imb_pipeline(
        StandardScaler(),
        PCA(n_components=PCA_number),
        SVC(C=1, kernel='rbf', random_state=42)
    )

pipeline_svc_2CV = make_imb_pipeline(
        StandardScaler(),
        PCA(n_components=PCA_number),
        SVC(C=1, kernel='rbf', random_state=42)
    )

pipeline_svc_2NLP = make_imb_pipeline(
        StandardScaler(),
        PCA(n_components=PCA_number),
        SVC(C=1, kernel='rbf', random_state=42)
    )

pipeline_svc_3CV = make_imb_pipeline(
        StandardScaler(),
        PCA(n_components=PCA_number),
        SVC(C=1, kernel='rbf', random_state=42)
    )

pipeline_svc_3NLP = make_imb_pipeline(
        StandardScaler(),
        PCA(n_components=PCA_number),
        SVC(C=1, kernel='rbf', random_state=42)
    )

pipeline_nbc_1CV = make_imb_pipeline(
        StandardScaler(),PCA(n_components=PCA_number),
        GaussianNB()
    )
pipeline_nbc_1NLP = make_imb_pipeline(
        StandardScaler(),PCA(n_components=PCA_number),
        GaussianNB()
    )

pipeline_nbc_2CV = make_imb_pipeline(
        StandardScaler(),PCA(n_components=PCA_number),
        GaussianNB()
    )

pipeline_nbc_2NLP = make_imb_pipeline(
        StandardScaler(),PCA(n_components=PCA_number),
    GaussianNB()
    )

pipeline_nbc_3CV = make_imb_pipeline(
        StandardScaler(),PCA(n_components=PCA_number),
        GaussianNB()
    )

pipeline_nbc_3NLP = make_imb_pipeline(
        StandardScaler(),PCA(n_components=PCA_number),
    GaussianNB()
    )

pipeline_lr_1CV = make_imb_pipeline(
        StandardScaler(),
    LogisticRegression(C=0.1, penalty='l1', solver='saga', class_weight='balanced', max_iter=500, n_jobs=-1)
)
pipeline_lr_1NLP = make_imb_pipeline(
        StandardScaler(),
    LogisticRegression(C=0.1, penalty='l1', solver='saga', class_weight='balanced', max_iter=500, n_jobs=-1)
)

pipeline_lr_2CV = make_imb_pipeline(
        StandardScaler(),
    LogisticRegression(C=0.1, penalty='l1', solver='saga', class_weight='balanced', max_iter=500, n_jobs=-1)
)

pipeline_lr_2NLP = make_imb_pipeline(
        StandardScaler(),
    LogisticRegression(C=0.1, penalty='l1', solver='saga', class_weight='balanced', max_iter=500, n_jobs=-1)
    )

pipeline_lr_3CV = make_imb_pipeline(
        StandardScaler(),
    LogisticRegression(C=0.1, penalty='l1', solver='saga', class_weight='balanced', max_iter=500, n_jobs=-1)
)

pipeline_lr_3NLP = make_imb_pipeline(
        StandardScaler(),
    LogisticRegression(C=0.1, penalty='l1', solver='saga', class_weight='balanced', max_iter=500, n_jobs=-1)
    )

# 100 times 10-fold CV
for experiment_num in range(100):
    print(f"\nExperiment {experiment_num + 1} / 100")
    results_list = []
    total_accuracy = []
    total_precision = []
    total_recall = []
    total_f1 = []

    # Initializes a list that stores all model predictions
    all_model_predictions = []

    # The outer 10-fold cross-validation divides the training set and the test set
    for fold_number, (train_index, test_index) in enumerate(outer_cv.split(features_task1_CV, low_self_esteem)):
        X_train_1_outer, X_test_1_outer = features_task1_CV.iloc[train_index], features_task1_CV.iloc[test_index]
        X_train_2_outer, X_test_2_outer = features_task1_NLP.iloc[train_index], features_task1_NLP.iloc[test_index]
        X_train_3_outer, X_test_3_outer = features_task2_CV.iloc[train_index], features_task2_CV.iloc[test_index]
        X_train_4_outer, X_test_4_outer = features_task2_NLP.iloc[train_index], features_task2_NLP.iloc[test_index]
        X_train_5_outer, X_test_5_outer = features_task3_CV.iloc[train_index], features_task3_CV.iloc[test_index]
        X_train_6_outer, X_test_6_outer = features_task3_NLP.iloc[train_index], features_task3_NLP.iloc[test_index]
        y_train_outer, y_test_outer = low_self_esteem.iloc[train_index], low_self_esteem.iloc[test_index]
        ids_test_outer = ids.iloc[test_index]

        models = [
            ('svc', [pipeline_svc_1CV, pipeline_svc_1NLP, pipeline_svc_2CV, pipeline_svc_2NLP, pipeline_svc_3CV, pipeline_svc_3NLP]),
            ('lda', [pipeline_lda_1CV, pipeline_lda_1NLP, pipeline_lda_2CV, pipeline_lda_2NLP, pipeline_lda_3CV, pipeline_lda_3NLP]),
            ('nbc', [pipeline_nbc_1CV, pipeline_nbc_1NLP, pipeline_nbc_2CV, pipeline_nbc_2NLP, pipeline_nbc_3CV, pipeline_nbc_3NLP]),
            ('lr', [pipeline_lr_1CV, pipeline_lr_1NLP, pipeline_lr_2CV, pipeline_lr_2NLP, pipeline_lr_3CV, pipeline_lr_3NLP]),
        ]

        # Stores an accuracy list of the individual models on each feature set (adjusted to a dictionary containing the list)
        inner_acc = {i: [] for i in range(6)}
        for inner_fold, (train_index_inner, test_index_inner) in enumerate(
                inner_cv.split(X_train_1_outer, y_train_outer)):

            # Split the inner training set and the verification set
            X_train_1_inner, X_val_1_inner = features_task1_CV.iloc[train_index_inner], features_task1_CV.iloc[
                test_index_inner]
            X_train_2_inner, X_val_2_inner = features_task1_NLP.iloc[train_index_inner], features_task1_NLP.iloc[
                test_index_inner]
            X_train_3_inner, X_val_3_inner = features_task2_CV.iloc[train_index_inner], features_task2_CV.iloc[
                test_index_inner]
            X_train_4_inner, X_val_4_inner = features_task2_NLP.iloc[train_index_inner], features_task2_NLP.iloc[
                test_index_inner]
            X_train_5_inner, X_val_5_inner = features_task3_CV.iloc[train_index_inner], features_task3_CV.iloc[
                test_index_inner]
            X_train_6_inner, X_val_6_inner = features_task3_NLP.iloc[train_index_inner], features_task3_NLP.iloc[
                test_index_inner]
            y_train_inner, y_val_inner = low_self_esteem.iloc[train_index_inner], low_self_esteem.iloc[
                test_index_inner]

            # The model for each feature set is trained and the accuracy is recorded
            feature_sets = [X_train_1_inner, X_train_2_inner, X_train_3_inner, X_train_4_inner, X_train_5_inner, X_train_6_inner]
            val_sets = [X_val_1_inner, X_val_2_inner, X_val_3_inner, X_val_4_inner, X_val_5_inner, X_val_6_inner]

            for model_name, pipelines in models:
                for k, X_train in enumerate(feature_sets):
                    pipeline = pipelines[k]
                    pipeline.fit(X_train, y_train_inner)
                    predictions = pipeline.predict(val_sets[k])
                    acc = accuracy_score(y_val_inner, predictions)
                    inner_acc[k].append((model_name, acc))

        # For each feature set, the two best-performing models are selected and the average accuracy is calculated
        best_models = {}
        average_accuracies = {}
        # For each feature set, the average accuracy of each model is calculated
        avg_accs = {i: {} for i in range(6)}

        # Calculate the average accuracy of each model on each feature set
        for feature_idx in range(6):
            # Iterate over all records of the current feature set and calculate the average accuracy of each model
            model_acc_dict = {}
            for model_name, acc in inner_acc[feature_idx]:
                if model_name not in model_acc_dict:
                    model_acc_dict[model_name] = []
                model_acc_dict[model_name].append(acc)

            # Average accuracy is calculated and stored
            for model_name, acc_list in model_acc_dict.items():
                avg_acc = np.mean(acc_list)
                avg_accs[feature_idx][model_name] = avg_acc

        # For each feature set, select the two models that perform best
        best_models = {}
        average_accuracies = {}

        for feature_idx in range(6):
            # Rank the average accuracy of the models and select the first two best models
            sorted_avg_accs = sorted(avg_accs[feature_idx].items(), key=lambda x: x[1], reverse=True)
            best_models[feature_idx] = sorted_avg_accs[:2]  # Two optimal models are selected on each feature set
            average_accuracies[feature_idx] = [acc for _, acc in sorted_avg_accs[:2]]

        # Output the optimal model for each feature set and its average accuracy
        selected_models = []  # The 8 best models used in the storage outer layer
        selected_accuracies = []  # The average accuracy of the 8 models used in the storage outer layer (as weights)

        # print("Selected best models and their average accuracies:")
        for feature_idx, models in best_models.items():
            for model_name, avg_acc in models:
                # print(f"Feature set {feature_idx + 1}, Model: {model_name}, Average Accuracy: {avg_acc}")
                selected_models.append((feature_idx, model_name))  # Store information about the best model (feature set number, model name)
                selected_accuracies.append(avg_acc)  # Store the average accuracy of the model

        sets = ['1CV','1NLP','2CV','2NLP','3CV','3NLP']
        outer_train = [X_train_1_outer, X_train_2_outer, X_train_3_outer, X_train_4_outer, X_train_5_outer, X_train_6_outer]
        test_sets = [X_test_1_outer, X_test_2_outer, X_test_3_outer, X_test_4_outer, X_test_5_outer, X_test_6_outer]

        # Outer training procedure
        predictions_all = []
        for feature_sets_idx, model in selected_models:
            # Training
            eval(f'pipeline_{model}_{sets[feature_sets_idx]}').fit(outer_train[feature_sets_idx], y_train_outer)
            # Prediction
            predictions = eval(f'pipeline_{model}_{sets[feature_sets_idx]}').predict(test_sets[feature_sets_idx])
            predictions_all.append(predictions)

        # Weighted voting
        final_predictions = []
        for idx in range(len(y_test_outer)):
            scores = {0: 0, 1: 0}
            for model, weight in zip(predictions_all, selected_accuracies):
                pred = model[idx]
                scores[pred] += weight
            final_predictions.append(max(scores, key=scores.get))

        # Calculates performance metrics on the test set
        test_accuracy = accuracy_score(y_test_outer, final_predictions)
        test_precision = precision_score(y_test_outer, final_predictions, pos_label=1, zero_division=0)
        test_recall = recall_score(y_test_outer, final_predictions, pos_label=1, zero_division=0)
        test_f1 = f1_score(y_test_outer, final_predictions, pos_label=1, zero_division=0)

        # Store results
        fold_results = pd.DataFrame({
                    'Experiment': experiment_num + 1,
                    'Fold': fold_number,
                    'ID': ids_test_outer,
                    'True Label': y_test_outer,
                    'Predicted Label': final_predictions,
                    'ACC': (final_predictions == y_test_outer).astype(int)
                })
        results_list.append(fold_results)

        # Accumulate the result of each fold
        total_accuracy.append(test_accuracy)
        total_precision.append(test_precision)
        total_recall.append(test_recall)
        total_f1.append(test_f1)

    # Print average and best results
    print(f"Average Accuracy: {np.mean(total_accuracy) * 100:.2f}%")
    print(f"Average Precision: {np.mean(total_precision) * 100:.2f}%")
    print(f"Average Recall: {np.mean(total_recall) * 100:.2f}%")
    print(f"Average F1: {np.mean(total_f1) * 100:.2f}%")

    # Concatenate all results
    all_results = pd.concat(results_list)
    all_results.to_excel(
            f'Task{task}\\ensemble_{modal}_result_{experiment_num + 1}.xlsx',
            index=False)

    # Record overall results
    overall_results.append({
            'Experiment': experiment_num + 1,
            'Average Accuracy': np.mean(total_accuracy),
            'Average Precision': np.mean(total_precision),
            'Average Recall': np.mean(total_recall),
            'Average F1': np.mean(total_f1)
        })

# Save the summary results of 100 experiments
overall_df = pd.DataFrame(overall_results)
overall_df.to_excel(f'ensemble_{modal}_task_{task}.xlsx',
            index=False)
