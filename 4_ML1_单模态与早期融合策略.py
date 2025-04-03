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


modals = ['AUs', 'Voice', 'Text', 'AUs+Voice', 'Voice+Text', 'AUs+Voice+Text']
models = ['SVC', 'NBC', 'LDA', 'LR']
date = '20250312'
inner_cv = StratifiedKFold(n_splits=10, shuffle=True,random_state=None)
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)


# 加载数据
data = pd.read_csv(f'E:')
self_esteem = data.iloc[:, -1]
low_self_esteem = (self_esteem < 31).astype(int)
ids = data.iloc[:, 0]
for modal in modals:
    if modal == 'AUs':
        features = data.iloc[:, 1:69]
    if modal == 'Voice':
        features = data.iloc[:, 69:157]
    if modal == 'Text':
        features = data.iloc[:, 157:-1]
    if modal == 'AUs+Voice':
        features = data.iloc[:, 1:157]
    if modal == 'Voice+Text':
        features = data.iloc[:, 69:-1]
    if modal == 'AUs+Voice+Text':
        features = data.iloc[:, 1:-1]

    for model in models:
        # 初始化存储所有实验结果的列表
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

        # 进行100次实验
        for study_num in range(100):
            print(f"\n study {study_num + 1} / 100")
            results_list = []
            total_accuracy = []
            total_precision = []
            total_recall = []
            total_f1 = []
            total_fold_number = 1
            acc_outer = []

            # 外层的10折交叉验证，划分训练集和测试集
            for fold_number, (train_index, test_index) in enumerate(outer_cv.split(features, low_self_esteem)):
                X_train_outer, X_test_outer = features.iloc[train_index], features.iloc[test_index]
                y_train_outer, y_test_outer = low_self_esteem[train_index], low_self_esteem[test_index]
                ids_test = ids.iloc[test_index]

                if model == 'SVC':
                    classifier = SVC(C=0.1, kernel='linear', class_weight='balanced', probability=True)
                if model == 'LR':
                    classifier = LogisticRegression(C=0.1, penalty='l2', class_weight='balanced', max_iter=1000, n_jobs=-1)
                if model == 'NBC':
                    classifier = GaussianNB()
                if model == 'LDA':
                    classifier = LinearDiscriminantAnalysis()

                final_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=0.9)),
                    ('model', classifier)
                ])

                best_model = final_pipeline.fit(X_train_outer, y_train_outer)

                # 在测试集上进行评估
                y_test_pred = best_model.predict(X_test_outer)
                test_accuracy = accuracy_score(y_test_outer, y_test_pred)
                test_precision = precision_score(y_test_outer, y_test_pred, pos_label=1, zero_division=0)
                test_recall = recall_score(y_test_outer, y_test_pred, pos_label=1, zero_division=0)
                test_f1 = f1_score(y_test_outer, y_test_pred, pos_label=1, zero_division=0)

                # 存储该折的结果
                result_df = pd.DataFrame({
                    'Fold': [total_fold_number],
                    'ID': [ids_test.tolist()],
                    'True Label': [y_test_outer.tolist()],
                    'Predicted Label': [y_test_pred.tolist()],
                    'ACC': [(y_test_outer == y_test_pred).astype(int).tolist()]
                })
                results_list.append(result_df)

                # 累积每个折叠的结果
                total_accuracy.append(test_accuracy)
                total_precision.append(test_precision)
                total_recall.append(test_recall)
                total_f1.append(test_f1)
                total_fold_number += 1

            # 打印平均指标和最佳结果
            print(f"Average Accuracy: {np.mean(total_accuracy) * 100:.2f}%")
            print(f"Average Precision: {np.mean(total_precision) * 100:.2f}%")
            print(f"Average Recall: {np.mean(total_recall) * 100:.2f}%")
            print(f"Average F1: {np.mean(total_f1) * 100:.2f}%")

            # Concatenate all results
            all_results = pd.concat(results_list)
            all_results.to_excel(f'E:', index=False)

            # 记录总体结果
            summary_df = pd.DataFrame({
                'Experiment': [study_num + 1],
                'Average Accuracy': [np.mean(total_accuracy)],
                'Average Precision': [np.mean(total_precision)],
                'Average Recall': [np.mean(total_recall)],
                'Average F1': [np.mean(total_f1)]
            })
            overall_results = pd.concat([overall_results, summary_df], ignore_index=True)

        # 保存30次实验的汇总结果
        overall_df = pd.DataFrame(overall_results)
        overall_df.to_excel(f'E:\\')
        print(classifier.get_params())
