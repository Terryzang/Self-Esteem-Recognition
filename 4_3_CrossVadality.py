from imblearn.pipeline import make_pipeline as make_imb_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score , accuracy_score, confusion_matrix
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

os.environ['JOBLIB_TEMP_FOLDER'] = 'D:\\joblib_temp'  # 指定一个无中文字符的临时路径
os.makedirs('D:\\joblib_temp', exist_ok=True)         # 如果不存在则创建


# todo 打印全部
'''打印np全部'''
np.set_printoptions(threshold=np.inf)
'''打印pd全部'''
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

date = '20250909'
name = '多模态'
weight = '朴素'

def proba_matrix_aligned(clf, X, global_classes=(0,1,2)):
    proba = clf.predict_proba(X)  # shape: (n_samples, n_present_classes)
    clf_classes = clf.classes_    # 模型内部的类别顺序
    out = np.zeros((proba.shape[0], len(global_classes)), dtype=float)

    for j, c in enumerate(global_classes):
        if c in clf_classes:
            idx = np.where(clf_classes == c)[0][0]
            out[:, j] = proba[:, idx]
        else:
            out[:, j] = 0.0  # 若某模型没见过这个类，概率列置0

    # 行归一化（避免全零导致除0）
    row_sum = out.sum(axis=1, keepdims=True)
    out = out / np.clip(row_sum, 1e-12, None)
    return out

# 交叉验证设置
outer_cv = StratifiedKFold(n_splits=10, shuffle=True)
train_df = pd.read_csv('old_all_feature_333.csv')   # N=211
test_df  = pd.read_csv('new_all_feature_333.csv')   # N=63
# 切分特征（与你之前保持一致的列段）
Xtr_AUs = train_df.iloc[:, 1:69]
Xtr_Voice = train_df.iloc[:, 69:157]
Xtr_Text = train_df.iloc[:, 157:-1]
ytr_raw = train_df.iloc[:, -1]         # RSES原始分

Xte_AUs = test_df.iloc[:, 1:69]
Xte_Voice = test_df.iloc[:, 69:157]
Xte_Text = test_df.iloc[:, 157:-1]
yte_raw = test_df.iloc[:, -1]
ids_te = test_df.iloc[:, 0]

tr_conditions = [
    (ytr_raw <= 28),
    (ytr_raw >= 30) & (ytr_raw <= 31),
    (ytr_raw >= 33)
]
te_conditions = [
    (yte_raw <= 28),
    (yte_raw >= 29) & (yte_raw <= 31),
    (yte_raw >= 32)
]
choices = [0, 1, 2]  # 低=0, 中=1, 高=2
tr_self_esteem = np.select(tr_conditions, choices, default=np.nan)
ytr = tr_self_esteem.astype(int)
te_self_esteem = np.select(te_conditions, choices, default=np.nan)
yte = te_self_esteem.astype(int)


# 初始化存储所有实验结果的列表
overall_results = []

# 创建使用Pipeline
pipeline_svc_AUs = make_imb_pipeline(
        StandardScaler(),
        # PCA(n_components=0.9),
        SVC(C=1, kernel ='linear',probability=True)
    )

pipeline_svc_Text = make_imb_pipeline(
    StandardScaler(),
    # PCA(n_components=0.95),
    SVC(C=1, kernel='rbf', probability=True)
    )

pipeline_svc_Voice = make_imb_pipeline(
        StandardScaler(),
        PCA(n_components=0.8),
        SVC(C=1, kernel='linear',probability=True)
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

results_list = []
total_accuracy = []
total_precision = []
total_recall = []
total_f1 = []

# 进行100次实验
if name == '多模态':
    number = 3
    AUs_svc = pipeline_svc_AUs.fit(Xtr_AUs, ytr)
    AUs_lr = pipeline_lr_AUs.fit(Xtr_AUs, ytr)
    AUs_nbc = pipeline_nbc_AUs.fit(Xtr_AUs, ytr)
    AUs_lda = pipeline_lda_AUs.fit(Xtr_AUs, ytr)
    Voice_svc = pipeline_svc_Voice.fit(Xtr_Voice, ytr)
    Voice_lr = pipeline_lr_Voice.fit(Xtr_Voice, ytr)
    Voice_nbc = pipeline_nbc_Voice.fit(Xtr_Voice, ytr)
    Voice_lda = pipeline_lda_Voice.fit(Xtr_Voice, ytr)
    Text_svc = pipeline_svc_Text.fit(Xtr_Text, ytr)
    Text_lr = pipeline_lr_Text.fit(Xtr_Text, ytr)
    Text_nbc = pipeline_nbc_Text.fit(Xtr_Text, ytr)
    Text_lda = pipeline_lda_Text.fit(Xtr_Text, ytr)

    global_classes = (0, 1, 2)

    # -- 模态内先等权平均（把每个模态的4个基模型合成1个概率矩阵）
    proba_list  = [
        proba_matrix_aligned(AUs_svc, Xte_AUs, global_classes),
        proba_matrix_aligned(AUs_lr, Xte_AUs, global_classes),
        proba_matrix_aligned(AUs_nbc, Xte_AUs, global_classes),
        proba_matrix_aligned(AUs_lda, Xte_AUs, global_classes),
        proba_matrix_aligned(Voice_svc, Xte_Voice, global_classes),
        proba_matrix_aligned(Voice_lr, Xte_Voice, global_classes),
        proba_matrix_aligned(Voice_nbc, Xte_Voice, global_classes),
        proba_matrix_aligned(Voice_lda, Xte_Voice, global_classes),
        proba_matrix_aligned(Text_svc, Xte_Text, global_classes),
        proba_matrix_aligned(Text_lr, Xte_Text, global_classes),
        proba_matrix_aligned(Text_nbc, Xte_Text, global_classes),
        proba_matrix_aligned(Text_lda, Xte_Text, global_classes),
    ]

    # 2) 叠成 (12, n_samples, 3)
    stacked = np.stack(proba_list, axis=0)

    if weight == '加权':
        w = np.array([
        0.379, 0.369, 0.359, 0.361,  # AUs: SVC, LR, NBC, LDA 的 F1
        0.384, 0.387, 0.387, 0.404,  # Voice: ...
        0.361, 0.341, 0.368, 0.345  # Text: ...
        ], dtype=float)

    elif weight == '朴素':
        w = np.array([
            1, 1, 1, 1,  # AUs: SVC, LR, NBC, LDA 的 F1
            1, 1, 1, 1,  # Voice: ...
            1, 1, 1, 1  # Text: ...
        ], dtype=float)

    w = w / (w.sum() + 1e-12)
    weighted_proba = np.tensordot(w, stacked, axes=(0, 0))  # -> (n_samples, 3)
    yte_pred = weighted_proba.argmax(axis=1)

    acc = accuracy_score(yte, yte_pred)
    prec = precision_score(yte, yte_pred, average='macro', zero_division=0)
    rec = recall_score(yte, yte_pred, average='macro', zero_division=0)
    f1 = f1_score(yte, yte_pred, average='macro', zero_division=0)
    cm = confusion_matrix(yte, yte_pred, labels=[0, 1, 2])

    print("\n=== Leave-out Test on NEW (63) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro): {rec:.4f}")
    print(f"F1 (macro): {f1:.4f}")
    print("Confusion Matrix [rows=true, cols=pred, order=0/1/2]:\n", cm)

    # 存储结果
    fold_results = pd.DataFrame({
        'ID': ids_te,
        'True Label': yte,
        'Predicted Label': yte_pred,
        'ACC': (yte == yte_pred).astype(int)
    })
    results_list.append(fold_results)

# Concatenate all results
all_results = pd.concat(results_list)
all_results.to_excel(
            f'{date}\\泛化测试_{name}_{weight}.xlsx',
            index=False)
