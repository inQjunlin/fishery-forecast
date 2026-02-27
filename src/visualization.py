# -*- coding: utf-8 -*-
"""
数据可视化工具：绘制 ROC/PR 曲线、特征重要性条形图等，方便论文展示。
依赖：matplotlib、scikit-learn
"""

import os
import json
from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

plt.style.use('ggplot')


def plot_roc_curve(y_true: Iterable[int] or np.ndarray, y_score: Iterable[float], save_path: str, title: str = None):
    y_true = np.asarray(list(y_true))
    y_score = np.asarray(list(y_score))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if title:
        plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return roc_auc


def plot_pr_curve(y_true: Iterable[int] or np.ndarray, y_score: Iterable[float], save_path: str, title: str = None):
    y_true = np.asarray(list(y_true))
    y_score = np.asarray(list(y_score))
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, color='b', lw=2, label=f'PR (AP = {ap:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if title:
        plt.title(title)
    plt.legend(loc='lower left')
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return ap


def plot_feature_importances(coef: Iterable[float], feature_names: Iterable[str], save_path: str, top_n: int = 20, title: str = None):
    coef = np.asarray(list(coef)).flatten()
    feature_names = list(feature_names)
    if coef.shape[0] != len(feature_names):
        # 尝试对齐，只取相等长度的前 N 品
        min_len = min(coef.shape[0], len(feature_names))
        coef = coef[:min_len]
        feature_names = feature_names[:min_len]
    idxs = np.argsort(np.abs(coef))[-top_n:][::-1]
    top_coef = coef[idxs]
    top_names = [feature_names[i] for i in idxs]
    plt.figure(figsize=(8, max(4, top_n * 0.4)))
    plt.barh(range(len(top_coef)), np.abs(top_coef), color='steelblue')
    plt.yticks(range(len(top_coef)), top_names)
    plt.xlabel('Absolute Coefficient')
    plt.title(title or f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
