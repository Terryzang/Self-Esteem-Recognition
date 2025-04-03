from imblearn.pipeline import make_pipeline as make_imb_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score , accuracy_score
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import RFE


name = '多模态'
modal = 'AUs+Voice'

# 交叉验证设置
outer_cv = StratifiedKFold(n_splits=10, shuffle=True)
inner_cv = StratifiedKFold(n_splits=10, shuffle=True,random_state=None)

data = pd.read_csv(f'E:')
features_AUs = data.iloc[:, 1:69]
features_Voice = data.iloc[:, 69:157]
features_Text = data.iloc[:, 157:-1]
self_esteem = data.iloc[:, -1]
low_self_esteem = (self_esteem < 31).astype(int)
ids = data.iloc[:, 0]

# 初始化存储所有实验结果的列表
overall_results = []

# 创建使用Pipeline
pipeline_svc_AUs = make_imb_pipeline(
        StandardScaler(),
        PCA(n_components=0.9),
        SVC(C=0.3, kernel ='linear',probability=True)
    )
pipeline_svc_Voice = make_imb_pipeline(
        StandardScaler(),
        PCA(n_components=0.1),
        SVC(C=1, kernel='linear',probability=True)
    )
pipeline_svc_Text = make_imb_pipeline(
        StandardScaler(),
        PCA(n_components=0.95),
        SVC(C=0.01, kernel='linear',probability=True)
    )

pipeline_lr_AUs = make_imb_pipeline(
        StandardScaler(),
        PCA(n_components=0.95),
        LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, n_jobs=-1)
)
pipeline_lr_Voice = make_imb_pipeline(
        StandardScaler(),
        PCA(n_components=0.3),
        LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, n_jobs=-1)
)
pipeline_lr_Text = make_imb_pipeline(
        StandardScaler(),
        PCA(n_components=0.9),
        LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, n_jobs=-1)
)

pipeline_nbc_AUs = make_imb_pipeline(
        StandardScaler(),
        GaussianNB()
    )
pipeline_nbc_Voice = make_imb_pipeline(
        StandardScaler(),
        GaussianNB()
    )
pipeline_nbc_Text = make_imb_pipeline(
        StandardScaler(),
        GaussianNB()
    )

pipeline_lda_AUs = make_imb_pipeline(
        StandardScaler(),
        PCA(n_components=0.95),
        LinearDiscriminantAnalysis()
    )
pipeline_lda_Voice = make_imb_pipeline(
        StandardScaler(),
        PCA(n_components=0.3),
        LinearDiscriminantAnalysis()
    )
pipeline_lda_Text = make_imb_pipeline(
        StandardScaler(),
        PCA(n_components=0.35),
        LinearDiscriminantAnalysis()
    )



# 进行100次实验
for experiment_num in range(100):
    print(f"\nExperiment {experiment_num + 1} / 100")
    results_list = []
    total_accuracy = []
    total_precision = []
    total_recall = []
    total_f1 = []

    # 初始化存储所有模型预测结果的列表
    all_model_predictions = []

    # 外层的10折交叉验证，划分训练集和测试集
    for fold_number, (train_index, test_index) in enumerate(outer_cv.split(features_AUs, low_self_esteem)):
        X_train_1_outer, X_test_1_outer = features_AUs.iloc[train_index], features_AUs.iloc[test_index]
        X_train_2_outer, X_test_2_outer = features_Voice.iloc[train_index], features_Voice.iloc[test_index]
        X_train_3_outer, X_test_3_outer = features_Text.iloc[train_index], features_Text.iloc[test_index]
        y_train_outer, y_test_outer = low_self_esteem.iloc[train_index], low_self_esteem.iloc[test_index]
        ids_test_outer = ids.iloc[test_index]

        if name == '多模态':
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

            # 获取各个模型的预测概率
            proba_AUs_svc = AUs_svc.predict_proba(X_test_1_outer)[:, 1]
            proba_AUs_lr = AUs_lr.predict_proba(X_test_1_outer)[:, 1]
            proba_AUs_nbc = AUs_nbc.predict_proba(X_test_1_outer)[:, 1]
            proba_AUs_lda = AUs_lda.predict_proba(X_test_1_outer)[:, 1]
            proba_Voice_svc = Voice_svc.predict_proba(X_test_2_outer)[:, 1]
            proba_Voice_lr = Voice_lr.predict_proba(X_test_2_outer)[:, 1]
            proba_Voice_nbc = Voice_nbc.predict_proba(X_test_2_outer)[:, 1]
            proba_Voice_lda = Voice_lda.predict_proba(X_test_2_outer)[:, 1]
            proba_Text_svc = Text_svc.predict_proba(X_test_3_outer)[:, 1]
            proba_Text_lr = Text_lr.predict_proba(X_test_3_outer)[:, 1]
            proba_Text_nbc = Text_nbc.predict_proba(X_test_3_outer)[:, 1]
            proba_Text_lda = Text_lda.predict_proba(X_test_3_outer)[:, 1]

            weighted_proba = (proba_AUs_svc +
                                  proba_AUs_lr +
                                  proba_AUs_nbc +
                                   proba_AUs_lda +
                                  proba_Voice_svc +
                                  proba_Voice_lr +
                                   proba_Voice_nbc +
                                  proba_Voice_lda +
                                  proba_Text_svc +
                                   proba_Text_lr +
                                   proba_Text_nbc +
                                  proba_Text_lda
                                  ) / 12


        if name == '双模态':
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


            # 获取各个模型的预测概率
            proba_AUs_svc = AUs_svc.predict_proba(X_test_1_outer)[:, 1]
            proba_AUs_lr = AUs_lr.predict_proba(X_test_1_outer)[:, 1]
            proba_AUs_nbc = AUs_nbc.predict_proba(X_test_1_outer)[:, 1]
            proba_AUs_lda = AUs_lda.predict_proba(X_test_1_outer)[:, 1]
            proba_Voice_svc = Voice_svc.predict_proba(X_test_2_outer)[:, 1]
            proba_Voice_lr = Voice_lr.predict_proba(X_test_2_outer)[:, 1]
            proba_Voice_nbc = Voice_nbc.predict_proba(X_test_2_outer)[:, 1]
            proba_Voice_lda = Voice_lda.predict_proba(X_test_2_outer)[:, 1]
            proba_Text_svc = Text_svc.predict_proba(X_test_3_outer)[:, 1]
            proba_Text_lr = Text_lr.predict_proba(X_test_3_outer)[:, 1]
            proba_Text_nbc = Text_nbc.predict_proba(X_test_3_outer)[:, 1]
            proba_Text_lda = Text_lda.predict_proba(X_test_3_outer)[:, 1]


            if modal == 'AUs+Voice':
                weighted_proba = (proba_AUs_svc +
                                  proba_AUs_lr +
                                  proba_AUs_nbc +
                                  proba_AUs_lda +
                                  proba_Voice_svc +
                                  proba_Voice_lr +
                                  proba_Voice_nbc +
                                  proba_Voice_lda
                                  ) / 8

            if modal == 'Voice+Text':
                weighted_proba = (
                                  proba_Voice_svc +
                                  proba_Voice_lr +
                                  proba_Voice_nbc +
                                  proba_Voice_lda +
                                  proba_Text_svc +
                                  proba_Text_lr +
                                  proba_Text_nbc +
                                  proba_Text_lda
                                  ) / 8

        # 最终分类决策（0=高自尊，1=低自尊）
        final_predictions = (weighted_proba >= 0.5).astype(int)

        # 计算测试集上的性能指标
        test_accuracy = accuracy_score(y_test_outer, final_predictions)
        test_precision = precision_score(y_test_outer, final_predictions, pos_label=1, zero_division=0)
        test_recall = recall_score(y_test_outer, final_predictions, pos_label=1, zero_division=0)
        test_f1 = f1_score(y_test_outer, final_predictions, pos_label=1, zero_division=0)

        # 存储结果
        fold_results = pd.DataFrame({
                    'Experiment': experiment_num + 1,
                    'Fold': fold_number,
                    'ID': ids_test_outer,
                    'True Label': y_test_outer,
                    'Predicted Label': final_predictions,
                    'ACC': (final_predictions == y_test_outer).astype(int)
                })
        results_list.append(fold_results)

        # 累积每个折叠的结果
        total_accuracy.append(test_accuracy)
        total_precision.append(test_precision)
        total_recall.append(test_recall)
        total_f1.append(test_f1)

    # 打印平均指标和最佳结果
    print(f"Average Accuracy: {np.mean(total_accuracy) * 100:.2f}%")
    print(f"Average Precision: {np.mean(total_precision) * 100:.2f}%")
    print(f"Average Recall: {np.mean(total_recall) * 100:.2f}%")
    print(f"Average F1: {np.mean(total_f1) * 100:.2f}%")

    # Concatenate all results
    all_results = pd.concat(results_list)
    all_results.to_excel(f'E:', index=False)

    # 记录总体结果
    overall_results.append({
            'Experiment': experiment_num + 1,
            'Average Accuracy': np.mean(total_accuracy),
            'Average Precision': np.mean(total_precision),
            'Average Recall': np.mean(total_recall),
            'Average F1': np.mean(total_f1)
        })

# 保存100次实验的汇总结果
overall_df = pd.DataFrame(overall_results)
overall_df.to_excel(f'E:', index=False)
