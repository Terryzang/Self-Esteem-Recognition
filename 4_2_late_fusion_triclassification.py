from imblearn.pipeline import make_pipeline as make_imb_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import RFE
import os
import joblib

os.environ['JOBLIB_TEMP_FOLDER'] = 'D:\\joblib_temp'  # Specify a temporary path without Chinese characters
os.makedirs('D:\\joblib_temp', exist_ok=True)         # Create it if it does not exist

name = 'multiple'

def proba_matrix_aligned(clf, X, global_classes=(0, 1, 2)):
    proba = clf.predict_proba(X)  # shape: (n_samples, n_present_classes)
    clf_classes = clf.classes_    # Class order inside the model
    out = np.zeros((proba.shape[0], len(global_classes)), dtype=float)

    for j, c in enumerate(global_classes):
        if c in clf_classes:
            idx = np.where(clf_classes == c)[0][0]
            out[:, j] = proba[:, idx]
        else:
            out[:, j] = 0.0  # If a model has never seen this class, set its probability column to 0

    # Row-wise normalization (avoid division by zero for all-zero rows)
    row_sum = out.sum(axis=1, keepdims=True)
    out = out / np.clip(row_sum, 1e-12, None)
    return out


# Cross-validation setting
outer_cv = StratifiedKFold(n_splits=10, shuffle=True)

data = pd.read_csv(f'all_feature.csv')
features_AUs = data.iloc[:, 1:69]
features_Voice = data.iloc[:, 69:157]
features_Text = data.iloc[:, 157:-1]
self_esteem = data.iloc[:, -1]
conditions = [
    (self_esteem <= 28),
    (self_esteem >= 30) & (self_esteem <= 31),
    (self_esteem >= 33)
]
choices = [0, 1, 2]  # low=0, medium=1, high=2
self_esteem_label = np.select(conditions, choices, default=np.nan)
y = self_esteem_label.astype(int)
ids = data.iloc[:, 0]

# Initialize a list to store all experiment results
overall_results = []

# Create pipelines
pipeline_svc_AUs = make_imb_pipeline(
    StandardScaler(),
    # PCA(n_components=0.9),
    SVC(C=1, kernel='linear', probability=True)
)

pipeline_svc_Text = make_imb_pipeline(
    StandardScaler(),
    # PCA(n_components=0.95),
    SVC(C=1, kernel='rbf', probability=True)
)

pipeline_svc_Voice = make_imb_pipeline(
    StandardScaler(),
    PCA(n_components=0.8),
    SVC(C=1, kernel='linear', probability=True)
)

pipeline_lr_AUs = make_imb_pipeline(
    StandardScaler(),
    PCA(n_components=0.95),
    LogisticRegression(C=1, class_weight='balanced', max_iter=1000, n_jobs=-1)
)

pipeline_lr_Text = make_imb_pipeline(
    StandardScaler(),
    PCA(n_components=0.95),
    LogisticRegression(C=0.01, class_weight='balanced', max_iter=1000, n_jobs=-1)
)

pipeline_lr_Voice = make_imb_pipeline(
    StandardScaler(),
    PCA(n_components=0.9),
    LogisticRegression(C=1, class_weight='balanced', max_iter=1000, n_jobs=-1)
)

pipeline_nbc_AUs = make_imb_pipeline(
    StandardScaler(),
    # PCA(n_components=0.95),
    GaussianNB()
)

pipeline_nbc_Text = make_imb_pipeline(
    StandardScaler(),
    # PCA(n_components=0.3),
    GaussianNB()
)

pipeline_nbc_Voice = make_imb_pipeline(
    StandardScaler(),
    # PCA(n_components=0.3),
    GaussianNB()
)

pipeline_lda_AUs = make_imb_pipeline(
    StandardScaler(),
    # PCA(n_components=0.95),
    LinearDiscriminantAnalysis()
)

pipeline_lda_Text = make_imb_pipeline(
    StandardScaler(),
    PCA(n_components=0.9),
    LinearDiscriminantAnalysis()
)

pipeline_lda_Voice = make_imb_pipeline(
    StandardScaler(),
    PCA(n_components=0.9),
    LinearDiscriminantAnalysis()
)

# Run 100 experiments
for experiment_num in range(100):
    print(f"\nExperiment {experiment_num + 1} / 100")
    results_list = []
    total_accuracy = []
    total_precision = []
    total_recall = []
    total_f1 = []

    # Initialize list to store predictions from all models (if needed)
    all_model_predictions = []

    # Outer 10-fold cross-validation: split into training and test sets
    for fold_number, (train_index, test_index) in enumerate(outer_cv.split(features_AUs, y)):
        X_train_1_outer, X_test_1_outer = features_AUs.iloc[train_index], features_AUs.iloc[test_index]
        X_train_2_outer, X_test_2_outer = features_Voice.iloc[train_index], features_Voice.iloc[test_index]
        X_train_3_outer, X_test_3_outer = features_Text.iloc[train_index], features_Text.iloc[test_index]
        y_train_outer, y_test_outer = y[train_index], y[test_index]
        ids_test_outer = ids.iloc[test_index]

        if name == 'multiple':
            number = 3
            AUs_svc = pipeline_svc_AUs.fit(X_train_1_outer, y_train_outer)
            AUs_lr = pipeline_lr_AUs.fit(X_train_1_outer, y_train_outer)
            AUs_nbc = pipeline_nbc_AUs.fit(X_train_1_outer, y_train_outer)
            AUs_lda = pipeline_lda_AUs.fit(X_train_1_outer, y_train_outer)
            Voice_svc = pipeline_svc_Voice.fit(X_train_2_outer, y_train_outer)
            Voice_lr = pipeline_lr_Voice.fit(X_train_2_outer, y_train_outer)
            Voice_nbc = pipeline_nbc_Voice.fit(X_train_2_outer, y_train_outer)
            Voice_lda = pipeline_lda_Voice.fit(X_train_2_outer, y_train_outer)
            Text_svc = pipeline_svc_Text.fit(X_train_3_outer, y_train_outer)
            Text_lr = pipeline_lr_Text.fit(X_train_3_outer, y_train_outer)
            Text_nbc = pipeline_nbc_Text.fit(X_train_3_outer, y_train_outer)
            Text_lda = pipeline_lda_Text.fit(X_train_3_outer, y_train_outer)

            # Get predicted probabilities from each model
            proba_AUs_svc = AUs_svc.predict_proba(X_test_1_outer)
            proba_AUs_lr = AUs_lr.predict_proba(X_test_1_outer)
            proba_AUs_nbc = AUs_nbc.predict_proba(X_test_1_outer)
            proba_AUs_lda = AUs_lda.predict_proba(X_test_1_outer)
            proba_Voice_svc = Voice_svc.predict_proba(X_test_2_outer)
            proba_Voice_lr = Voice_lr.predict_proba(X_test_2_outer)
            proba_Voice_nbc = Voice_nbc.predict_proba(X_test_2_outer)
            proba_Voice_lda = Voice_lda.predict_proba(X_test_2_outer)
            proba_Text_svc = Text_svc.predict_proba(X_test_3_outer)
            proba_Text_lr = Text_lr.predict_proba(X_test_3_outer)
            proba_Text_nbc = Text_nbc.predict_proba(X_test_3_outer)
            proba_Text_lda = Text_lda.predict_proba(X_test_3_outer)

            global_classes = (0, 1, 2)
            proba_list = [
                proba_matrix_aligned(AUs_svc, X_test_1_outer, global_classes),
                proba_matrix_aligned(AUs_lr, X_test_1_outer, global_classes),
                proba_matrix_aligned(AUs_nbc, X_test_1_outer, global_classes),
                proba_matrix_aligned(AUs_lda, X_test_1_outer, global_classes),

                proba_matrix_aligned(Voice_svc, X_test_2_outer, global_classes),
                proba_matrix_aligned(Voice_lr, X_test_2_outer, global_classes),
                proba_matrix_aligned(Voice_nbc, X_test_2_outer, global_classes),
                proba_matrix_aligned(Voice_lda, X_test_2_outer, global_classes),

                proba_matrix_aligned(Text_svc, X_test_3_outer, global_classes),
                proba_matrix_aligned(Text_lr, X_test_3_outer, global_classes),
                proba_matrix_aligned(Text_nbc, X_test_3_outer, global_classes),
                proba_matrix_aligned(Text_lda, X_test_3_outer, global_classes),
            ]

            # Naive (equal-weight) soft voting: average along axis=0 for (12, n_samples, 3)
            stacked = np.stack(proba_list, axis=0)   # -> (12, n_samples, 3)
            weighted_proba = stacked.mean(axis=0)    # -> (n_samples, 3)

            # Final decision (mutually exclusive three-class): take the argmax per row
            final_predictions = weighted_proba.argmax(axis=1)  # shape (n_samples,)

            w12 = w12 / (w12.sum() + 1e-12)
            weighted_proba = np.tensordot(w12, stacked, axes=(0, 0))  # -> (n_samples, 3)
            # Final decision (mutually exclusive three-class): take the argmax per row
            final_predictions = weighted_proba.argmax(axis=1)  # shape (n_samples,)

        elif name == 'audio-visual':
            number = 1
            AUs_svc = pipeline_svc_AUs.fit(X_train_1_outer, y_train_outer)
            AUs_lr = pipeline_lr_AUs.fit(X_train_1_outer, y_train_outer)
            AUs_nbc = pipeline_nbc_AUs.fit(X_train_1_outer, y_train_outer)
            AUs_lda = pipeline_lda_AUs.fit(X_train_1_outer, y_train_outer)
            Voice_svc = pipeline_svc_Voice.fit(X_train_2_outer, y_train_outer)
            Voice_lr = pipeline_lr_Voice.fit(X_train_2_outer, y_train_outer)
            Voice_nbc = pipeline_nbc_Voice.fit(X_train_2_outer, y_train_outer)
            Voice_lda = pipeline_lda_Voice.fit(X_train_2_outer, y_train_outer)

            # Get predicted probabilities from each model
            proba_AUs_svc = AUs_svc.predict_proba(X_test_1_outer)
            proba_AUs_lr = AUs_lr.predict_proba(X_test_1_outer)
            proba_AUs_nbc = AUs_nbc.predict_proba(X_test_1_outer)
            proba_AUs_lda = AUs_lda.predict_proba(X_test_1_outer)
            proba_Voice_svc = Voice_svc.predict_proba(X_test_2_outer)
            proba_Voice_lr = Voice_lr.predict_proba(X_test_2_outer)
            proba_Voice_nbc = Voice_nbc.predict_proba(X_test_2_outer)
            proba_Voice_lda = Voice_lda.predict_proba(X_test_2_outer)

            global_classes = (0, 1, 2)
            proba_list = [
                proba_matrix_aligned(AUs_svc, X_test_1_outer, global_classes),
                proba_matrix_aligned(AUs_lr, X_test_1_outer, global_classes),
                proba_matrix_aligned(AUs_nbc, X_test_1_outer, global_classes),
                proba_matrix_aligned(AUs_lda, X_test_1_outer, global_classes),

                proba_matrix_aligned(Voice_svc, X_test_2_outer, global_classes),
                proba_matrix_aligned(Voice_lr, X_test_2_outer, global_classes),
                proba_matrix_aligned(Voice_nbc, X_test_2_outer, global_classes),
                proba_matrix_aligned(Voice_lda, X_test_2_outer, global_classes)
            ]

            # Naive (equal-weight) soft voting: average along axis=0 for (12, n_samples, 3)
            stacked = np.stack(proba_list, axis=0)   # -> (12, n_samples, 3)
            weighted_proba = stacked.mean(axis=0)    # -> (n_samples, 3)

            # Final decision (mutually exclusive three-class): take the argmax per row
            final_predictions = weighted_proba.argmax(axis=1)  # shape (n_samples,)

        elif name == 'audio-text':
            number = 2
            Voice_svc = pipeline_svc_Voice.fit(X_train_2_outer, y_train_outer)
            Voice_lr = pipeline_lr_Voice.fit(X_train_2_outer, y_train_outer)
            Voice_nbc = pipeline_nbc_Voice.fit(X_train_2_outer, y_train_outer)
            Voice_lda = pipeline_lda_Voice.fit(X_train_2_outer, y_train_outer)
            Text_svc = pipeline_svc_Text.fit(X_train_3_outer, y_train_outer)
            Text_lr = pipeline_lr_Text.fit(X_train_3_outer, y_train_outer)
            Text_nbc = pipeline_nbc_Text.fit(X_train_3_outer, y_train_outer)
            Text_lda = pipeline_lda_Text.fit(X_train_3_outer, y_train_outer)

            # Get predicted probabilities from each model
            proba_Voice_svc = Voice_svc.predict_proba(X_test_2_outer)
            proba_Voice_lr = Voice_lr.predict_proba(X_test_2_outer)
            proba_Voice_nbc = Voice_nbc.predict_proba(X_test_2_outer)
            proba_Voice_lda = Voice_lda.predict_proba(X_test_2_outer)
            proba_Text_svc = Text_svc.predict_proba(X_test_3_outer)
            proba_Text_lr = Text_lr.predict_proba(X_test_3_outer)
            proba_Text_nbc = Text_nbc.predict_proba(X_test_3_outer)
            proba_Text_lda = Text_lda.predict_proba(X_test_3_outer)

            global_classes = (0, 1, 2)
            proba_list = [
                proba_matrix_aligned(Voice_svc, X_test_2_outer, global_classes),
                proba_matrix_aligned(Voice_lr, X_test_2_outer, global_classes),
                proba_matrix_aligned(Voice_nbc, X_test_2_outer, global_classes),
                proba_matrix_aligned(Voice_lda, X_test_2_outer, global_classes),

                proba_matrix_aligned(Text_svc, X_test_3_outer, global_classes),
                proba_matrix_aligned(Text_lr, X_test_3_outer, global_classes),
                proba_matrix_aligned(Text_nbc, X_test_3_outer, global_classes),
                proba_matrix_aligned(Text_lda, X_test_3_outer, global_classes)
            ]

            # Naive (equal-weight) soft voting: average along axis=0 for (12, n_samples, 3)
            stacked = np.stack(proba_list, axis=0)   # -> (12, n_samples, 3)
            weighted_proba = stacked.mean(axis=0)    # -> (n_samples, 3)

            # Final decision (mutually exclusive three-class): take the argmax per row
            final_predictions = weighted_proba.argmax(axis=1)  # shape (n_samples,)

        # Compute performance metrics on the test set
        test_accuracy = accuracy_score(y_test_outer, final_predictions)
        test_precision = precision_score(y_test_outer, final_predictions, average='macro', zero_division=0)
        test_recall = recall_score(y_test_outer, final_predictions, average='macro', zero_division=0)
        test_f1 = f1_score(y_test_outer, final_predictions, average='macro', zero_division=0)

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

        # Accumulate results for each fold
        total_accuracy.append(test_accuracy)
        total_precision.append(test_precision)
        total_recall.append(test_recall)
        total_f1.append(test_f1)

    # Print average metrics
    print(f"Average Accuracy: {np.mean(total_accuracy) * 100:.2f}%")
    print(f"Average Precision: {np.mean(total_precision) * 100:.2f}%")
    print(f"Average Recall: {np.mean(total_recall) * 100:.2f}%")
    print(f"Average F1: {np.mean(total_f1) * 100:.2f}%")

    # Concatenate all results
    all_results = pd.concat(results_list)
    all_results.to_excel(
        f'/{number}_{name}_{weight}_{experiment_num + 1}.xlsx',
        index=False)

    # Record overall results
    overall_results.append({
        'Experiment': experiment_num + 1,
        'Average Accuracy': np.mean(total_accuracy),
        'Average Precision': np.mean(total_precision),
        'Average Recall': np.mean(total_recall),
        'Average F1': np.mean(total_f1)
    })

        # Save summary results of 100 experiments
overall_df = pd.DataFrame(overall_results)
overall_df.to_excel(
    f'/{name}_task_{task}.xlsx', index=False)
