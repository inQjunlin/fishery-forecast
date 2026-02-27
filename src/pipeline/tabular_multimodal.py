# -*- coding: utf-8 -*-
"""
Tabular多模态训练管线：将数值型特征与分类型特征融合，提供三种方向：
- baseline_numeric: 仅数值特征、简单 LR/逻辑回归
- multimodal_fusion: 数值+分类型特征并行编码后融合
- cross_modal_attention: 两个分支的概率输出作为输入，训练一个简单的元学习层进行融合
"""

import os
import json
import numpy as np
import joblib
from typing import List

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd


def _split_features(df: pd.DataFrame, label_col: str = 'label'):
    # 删除不参与建模的字段
    if 'FILENAME' in df.columns:
        df = df.drop(columns=['FILENAME'])
    # 目标
    y = df[label_col].values
    X = df.drop(columns=[label_col])
    # 数值型特征与分类型特征的区分
    numeric_features: List[str] = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    categorical_features: List[str] = [col for col in X.columns if col not in numeric_features]
    return X, y, numeric_features, categorical_features


def train_baseline_numeric(df: pd.DataFrame, out_dir: str = 'models', label_col: str = 'label', test_size: float = 0.2, random_state: int = 42):
    X, y, numeric_features, _ = _split_features(df, label_col=label_col)
    X_num = X[numeric_features]
    # 构建 Pipeline: 标准化 + 逻辑回归
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, solver='lbfgs'))
    ])
    X_tr, X_te, y_tr, y_te = train_test_split(X_num, y, test_size=test_size, random_state=random_state, stratify=y)
    pipe.fit(X_tr, y_tr)
    y_score = pipe.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_score)
    acc = accuracy_score(y_te, (y_score >= 0.5).astype(int))
    f1 = f1_score(y_te, (y_score >= 0.5).astype(int))
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'baseline_numeric_model.pkl')
    joblib.dump({'model': pipe, 'features': numeric_features}, model_path)
    metrics = {'auc': float(auc), 'accuracy': float(acc), 'f1': float(f1)}
    # 计算并保存 ROC/PR 数据，便于后续可视化
    roc_data = {
        'y_true': y_te.tolist(),
        'y_score': y_score.tolist()
    }
    precision, recall, thresholds = precision_recall_curve(y_te, y_score)
    ap = average_precision_score(y_te, y_score)
    pr_data = {
        'y_true': y_te.tolist(),
        'y_score': y_score.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'thresholds': thresholds.tolist(),
        'average_precision': float(ap)
    }
    with open(os.path.join(out_dir, 'baseline_numeric_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return model_path, metrics, roc_data, pr_data


def train_tabular_multimodal(df: pd.DataFrame, out_dir: str = 'models', label_col: str = 'label', test_size: float = 0.2, random_state: int = 42):
    X, y, numeric_features, categorical_features = _split_features(df, label_col=label_col)
    if len(numeric_features) == 0:
        raise ValueError("没有可用的数值特征用于基线数模态训练")
    if len(categorical_features) == 0:
        # 没有分类型特征时，退回到数值特征模型
        return train_baseline_numeric(df, out_dir=out_dir, label_col=label_col, test_size=test_size, random_state=random_state)
    X_num = X[numeric_features]
    X_cat = X[categorical_features]
    # 预处理：数值特征标准化，分类型特征独热编码
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    # 模型：逻辑回归（高维稀疏特征可扩展性好）
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', clf)])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    pipe.fit(X_tr, y_tr)
    y_score = pipe.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_score)
    acc = accuracy_score(y_te, (y_score >= 0.5).astype(int))
    f1 = f1_score(y_te, (y_score >= 0.5).astype(int))
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'tabular_multimodal_model.pkl')
    joblib.dump({'model': pipe, 'numeric_features': numeric_features, 'categorical_features': categorical_features}, model_path)
    metrics = {'auc': float(auc), 'accuracy': float(acc), 'f1': float(f1)}
    roc_data = {'y_true': y_te.tolist(), 'y_score': y_score.tolist()}
    precision, recall, thresholds = precision_recall_curve(y_te, y_score)
    ap = average_precision_score(y_te, y_score)
    pr_data = {
        'y_true': y_te.tolist(),
        'y_score': y_score.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'thresholds': thresholds.tolist(),
        'average_precision': float(ap)
    }
    with open(os.path.join(out_dir, 'tabular_multimodal_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return model_path, metrics, roc_data, pr_data


def train_cross_modal_attention(df: pd.DataFrame, out_dir: str = 'models', label_col: str = 'label', test_size: float = 0.2, random_state: int = 42):
    """Cross-Modal Attention with Out-of-Fold (OOF) stacking to avoid data leakage.
    Implement a proper stacking scheme:
      - Use 5-fold StratifiedKFold on the training portion to generate OOF predictions for both bases.
      - Train meta-model on OOF predictions (training part only).
      - Train base models on full training data and generate test predictions for final scoring.
    """
    X, y, numeric_features, categorical_features = _split_features(df, label_col=label_col)
    if len(numeric_features) == 0 or len(categorical_features) == 0:
        return train_tabular_multimodal(df, out_dir, label_col=label_col, test_size=test_size, random_state=random_state)
    X_num = X[numeric_features]
    X_cat = X[categorical_features]
    # 1) 拆分训练集和测试集，保留测试集用于最终评估
    X_all = pd.concat([X_num, X_cat], axis=1)
    X_tr_full, X_te, y_tr_full, y_te = train_test_split(X_all, y, test_size=test_size, random_state=random_state, stratify=y)
    X_tr_full_num = X_tr_full[numeric_features]
    X_tr_full_cat = X_tr_full[categorical_features]
    # 2) 基模型的 OO F 训练
    num_pipeline = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, n_jobs=-1, solver='lbfgs'))])
    cat_pipeline = Pipeline([('pre', OneHotEncoder(handle_unknown='ignore')), ('clf', LogisticRegression(max_iter=1000, n_jobs=-1, solver='lbfgs'))])
    oof_num_proba = np.zeros(len(y_tr_full))
    oof_cat_proba = np.zeros(len(y_tr_full))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for train_idx, val_idx in skf.split(X_tr_full, y_tr_full):
        X_num_train = X_tr_full_num.iloc[train_idx]
        X_num_val = X_tr_full_num.iloc[val_idx]
        y_train = y_tr_full[train_idx]
        num_model_fold = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, n_jobs=-1, solver='lbfgs'))])
        num_model_fold.fit(X_num_train, y_train)
        oof_num_proba[val_idx] = num_model_fold.predict_proba(X_num_val)[:, 1]
        X_cat_train = X_tr_full_cat.iloc[train_idx]
        X_cat_val = X_tr_full_cat.iloc[val_idx]
        cat_model_fold = Pipeline([('pre', OneHotEncoder(handle_unknown='ignore')), ('clf', LogisticRegression(max_iter=1000, n_jobs=-1, solver='lbfgs'))])
        cat_model_fold.fit(X_cat_train, y_train)
        oof_cat_proba[val_idx] = cat_model_fold.predict_proba(X_cat_val)[:, 1]
    # 3) 训练元模型
    X_meta_train = np.vstack([oof_num_proba, oof_cat_proba]).T
    final_model = LogisticRegression(max_iter=1000, n_jobs=-1)
    final_model.fit(X_meta_train, y_tr_full)
    # 4) 使用全量数据训练基模型，用于预测测试集
    num_model_full = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, n_jobs=-1, solver='lbfgs'))])
    num_model_full.fit(X_tr_full_num, y_tr_full)
    cat_model_full = Pipeline([('pre', OneHotEncoder(handle_unknown='ignore')), ('clf', LogisticRegression(max_iter=1000, n_jobs=-1, solver='lbfgs'))])
    cat_model_full.fit(X_tr_full_cat, y_tr_full)
    # 5) 测试集预测
    X_te_num = X_te[numeric_features]
    X_te_cat = X_te[categorical_features]
    y_num_proba_test = num_model_full.predict_proba(X_te_num)[:, 1]
    y_cat_proba_test = cat_model_full.predict_proba(X_te_cat)[:, 1]
    X_meta_test = np.vstack([y_num_proba_test, y_cat_proba_test]).T
    y_score = final_model.predict_proba(X_meta_test)[:, 1]
    auc = roc_auc_score(y_te, y_score)
    acc = accuracy_score(y_te, (y_score >= 0.5).astype(int))
    f1 = f1_score(y_te, (y_score >= 0.5).astype(int))
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'cross_modal_attention_model.pkl')
    joblib.dump({'num_model_full': num_model_full, 'cat_model_full': cat_model_full, 'final_model': final_model}, model_path)
    metrics = {'auc': float(auc), 'accuracy': float(acc), 'f1': float(f1)}
    roc_data = {'y_true': y_te.tolist(), 'y_score': y_score.tolist()}
    precision, recall, thresholds = precision_recall_curve(y_te, y_score)
    ap = average_precision_score(y_te, y_score)
    pr_data = {
        'y_true': y_te.tolist(),
        'y_score': y_score.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'thresholds': thresholds.tolist(),
        'average_precision': float(ap)
    }
    with open(os.path.join(out_dir, 'cross_modal_attention_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return model_path, metrics, roc_data, pr_data
