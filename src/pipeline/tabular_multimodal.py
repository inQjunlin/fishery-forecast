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
    """简单的Cross-Modal Attention骨架：分别训练数值模型与分类模型，然后用一个二元特征集训练最终模型"""
    # 先训练数值与分类两条分支
    X, y, numeric_features, categorical_features = _split_features(df, label_col=label_col)
    if len(numeric_features) == 0 or len(categorical_features) == 0:
        return train_tabular_multimodal(df, out_dir, label_col=label_col, test_size=test_size, random_state=random_state)
    X_num = X[numeric_features]
    X_cat = X[categorical_features]
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    num_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, n_jobs=-1, solver='lbfgs'))
    ])
    cat_pipeline = Pipeline([
        ('pre', OneHotEncoder(handle_unknown='ignore')),  # 仅编码，不做缩放
        ('clf', LogisticRegression(max_iter=1000, n_jobs=-1, solver='lbfgs'))
    ])
    X_tr, X_te, y_tr, y_te = train_test_split(pd.concat([X_num, X_cat], axis=1), y, test_size=test_size, random_state=random_state, stratify=y)
    # 训练两个分支（注意：这里为骨架演示，实际应分别对 X_num/X_cat 建模，简化实现如下）
    num_model = num_pipeline.fit(X_tr[numeric_features], y_tr)
    cat_model = cat_pipeline.fit(X_tr[categorical_features], y_tr)
    y_num_proba = num_model.predict_proba(X_te[numeric_features])[:, 1]
    y_cat_proba = cat_model.predict_proba(X_te[categorical_features])[:, 1]
    X_stack = np.vstack([y_num_proba, y_cat_proba]).T
    final_model = LogisticRegression(max_iter=1000, n_jobs=-1)
    final_model.fit(X_stack, y_te)
    y_score = final_model.predict_proba(X_stack)[:, 1]
    auc = roc_auc_score(y_te, y_score)
    acc = accuracy_score(y_te, (y_score >= 0.5).astype(int))
    f1 = f1_score(y_te, (y_score >= 0.5).astype(int))
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'cross_modal_attention_model.pkl')
    joblib.dump({'num_model': num_model, 'cat_model': cat_model, 'final_model': final_model}, model_path)
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
