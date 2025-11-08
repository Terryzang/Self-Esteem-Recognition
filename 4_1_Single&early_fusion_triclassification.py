from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

models = ['LR']
modals = ['AUs+Voice+Text']
c = 0.01
pca = 0.99
kernel = 'linear'
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

# Load data
data = pd.read_csv(f'all_feature.csv')
self_esteem = data.iloc[:, -1]
print(self_esteem.value_counts().sort_index())

# Three-class split: low (≤28), medium (30–31), high (≥33); others are removed
conditions = [
    (self_esteem <= 28),
    (self_esteem >= 30) & (self_esteem <= 31),
    (self_esteem >= 33)
]
choices = [0, 1, 2]  # low=0, medium=1, high=2
self_esteem_label = np.select(conditions, choices, default=np.nan)
ids = data.iloc[:, 0]
print(self_esteem_label)

for modal in modals:
    if modal == 'AUs':
        features = data.iloc[:, 1:69]
        name = '1 single modality'
    if modal == 'Voice':
        features = data.iloc[:, 69:157]
        name = '1 single modality'
    if modal == 'Text':
        features = data.iloc[:, 157:-1]
        name = '1 single modality'
    if modal == 'AUs+Voice':
        features = data.iloc[:, 1:157]
        name = '2 early fusion of two modalities'
    if modal == 'Voice+Text':
        features = data.iloc[:, 69:-1]
        name = '2 early fusion of two modalities'
    if modal == 'AUs+Voice+Text':
        features = data.iloc[:, 1:-1]
        name = '3 early fusion of three modalities'

    for model in models:
        # Initialize the DataFrame to store all experiment results
        overall_results = pd.DataFrame(
            columns=['Experiment', 'Average Accuracy', 'Average Precision', 'Average Recall', 'Average F1'])
        if model == 'LR':
            selection = 'LASSO'
        if model == 'SVC':
            selection = 'SVC'
        if model == 'NBC':
            selection = 'NBC'
        if model == 'LDA':
            selection = 'LDA'

        # Run 100 experiments
        for study_num in range(100):
            print(f"\n study {study_num + 1} / 100")
            results_list = []
            total_accuracy = []
            total_precision = []
            total_recall = []
            total_f1 = []
            total_fold_number = 1
            acc_outer = []

            # 10-fold cross-validation: split into training and test sets
            for fold_number, (train_index, test_index) in enumerate(outer_cv.split(features, self_esteem_label)):
                X_train_outer, X_test_outer = features.iloc[train_index], features.iloc[test_index]
                y_train_outer, y_test_outer = self_esteem_label[train_index], self_esteem_label[test_index]
                ids_test = ids.iloc[test_index]

                if model == 'SVC':
                    classifier = SVC(C=c, kernel=kernel, class_weight='balanced', probability=True)

                elif model == 'LR':
                    classifier = LogisticRegression(C=c, penalty='l2', class_weight='balanced',
                                                    max_iter=1000, n_jobs=-1)
                elif model == 'NBC':
                    classifier = GaussianNB()

                elif model == 'LDA':
                    classifier = LinearDiscriminantAnalysis()

                else:
                    raise ValueError(f"Unknown model: {model}")

                if isinstance(pca, (int, float)):
                    final_pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('pca', PCA(n_components=pca)),
                        ('model', classifier)
                    ])

                elif (pca is None) or (isinstance(pca, str) and pca.lower() == 'none'):
                    final_pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('model', classifier)
                    ])

                else:
                    raise ValueError(f"Invalid pca specification: {pca!r}")

                best_model = final_pipeline.fit(X_train_outer, y_train_outer)

                # Evaluate performance on the training set
                y_train_pred = best_model.predict(X_train_outer)
                train_accuracy = accuracy_score(y_train_outer, y_train_pred)
                train_precision = precision_score(y_train_outer, y_train_pred, average='macro', zero_division=0)
                train_recall = recall_score(y_train_outer, y_train_pred, average='macro', zero_division=0)
                train_f1 = f1_score(y_train_outer, y_train_pred, average='macro', zero_division=0)

                # Evaluate performance on the test set
                y_test_pred = best_model.predict(X_test_outer)
                test_accuracy = accuracy_score(y_test_outer, y_test_pred)
                test_precision = precision_score(y_test_outer, y_test_pred, average='macro', zero_division=0)
                test_recall = recall_score(y_test_outer, y_test_pred, average='macro', zero_division=0)
                test_f1 = f1_score(y_test_outer, y_test_pred, average='macro', zero_division=0)

                # Print results
                print(
                    f"Training - Accuracy: {train_accuracy:2f}, Precision: {train_precision:2f}, Recall: {train_recall:2f}, F1 Score: {train_f1:2f}")
                print(
                    f"Testing - Accuracy: {test_accuracy:2f}, Precision: {test_precision:2f}, Recall: {test_recall:2f}, F1 Score: {test_f1:2f}")

                # Store results for this fold
                result_df = pd.DataFrame({
                    'Fold': total_fold_number,
                    'ID': ids_test,
                    'True Label': y_test_outer,
                    'Predicted Label': y_test_pred,
                    'ACC': (y_test_outer == y_test_pred).astype(int)
                })
                results_list.append(result_df)

                # Accumulate metrics for this fold
                total_accuracy.append(test_accuracy)
                total_precision.append(test_precision)
                total_recall.append(test_recall)
                total_f1.append(test_f1)
                total_fold_number += 1

            # Print average metrics
            print(f"Average Accuracy: {np.mean(total_accuracy) * 100:.2f}%")
            print(f"Average Precision: {np.mean(total_precision) * 100:.2f}%")
            print(f"Average Recall: {np.mean(total_recall) * 100:.2f}%")
            print(f"Average F1: {np.mean(total_f1) * 100:.2f}%")

            # Concatenate all fold results
            all_results = pd.concat(results_list)
            all_results.to_excel(
                f'E:/{model}_{study_num + 1}_result.xlsx',
                index=False)

            # Record summary results for this experiment
            summary_df = pd.DataFrame({
                'Experiment': [study_num + 1],
                'Average Accuracy': [np.mean(total_accuracy)],
                'Average Precision': [np.mean(total_precision)],
                'Average Recall': [np.mean(total_recall)],
                'Average F1': [np.mean(total_f1)]
            })
            overall_results = pd.concat([overall_results, summary_df], ignore_index=True)

        # Save the summary of 100 experiments
        overall_df = pd.DataFrame(overall_results)
        overall_df.to_excel(f'/{modal}_{model}_pca{pca}_c{c}.xlsx',
                            index=False)
        print(classifier.get_params())
