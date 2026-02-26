#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练管线：从数据加载、特征提取、到早期融合训练与评估的一体化流程
实现要点:
- 支持从 data/raw/ 读取 CSV 数据，自动探测 url/text/label 列
- 提取 URL 结构特征与文本嵌入特征
- 使用早期融合（URL特征 + 文本嵌入）进行二分类训练与评估
- 训练完成后将模型保存到 models/ 目录
"""

import os
import pickle
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

import argparse

from src.data_loader import load_dataset
from src.feature_extraction import extract_url_features, vectorize_texts
from src.fusion.early_fusion import train_early_fusion_from_slices

def _detect_columns(df):
    url_col = None
    for c in df.columns:
        if str(c).lower() in {"url", "link", "uri", "href"}:
            url_col = c
            break
    if url_col is None:
        raise ValueError("数据中未发现 URL 列，请确保包含 'url'、'link'、'uri' 等列名")
    text_col = None
    for c in df.columns:
        if str(c).lower() in {"text", "body", "content", "title", "message"}:
            text_col = c
            break
    label_col = None
    for c in df.columns:
        if str(c).lower() in {"label", "target", "y"}:
            label_col = c
            break
    if label_col is None:
        if "label" in df.columns:
            label_col = "label"
    if label_col is None:
        raise ValueError("数据中缺少标签列（label/target/y）")
    return url_col, text_col, label_col

def _prepare_features_from_df(df, url_col, text_col, label_col):
    # URL 特征
    urls = df[url_col].fillna('').astype(str).tolist()
    url_features_list = [extract_url_features(u) for u in urls]
    url_features_df = None
    if url_features_list:
        import pandas as pd
        url_features_df = pd.DataFrame(url_features_list).fillna(0)
        X_url = csr_matrix(url_features_df.values)
    else:
        X_url = csr_matrix((len(urls), 0))
    # 文本特征
    if text_col and text_col in df.columns:
        texts = df[text_col].fillna('').astype(str).tolist()
        if any(len(t.strip()) > 0 for t in texts):
            X_text, text_vec = vectorize_texts(texts)
        else:
            X_text = csr_matrix((len(texts), 0))
            text_vec = None
    else:
        X_text = csr_matrix((len(urls), 0))
        text_vec = None
    # 融合特征
    if X_text.shape[1] > 0:
        X_fused = hstack([X_url, X_text])
    else:
        X_fused = X_url
    # 标签
    if label_col in df.columns:
        y = df[label_col].astype(int).values
    else:
        raise ValueError(f"数据中缺少标签列 '{label_col}'")
    url_feature_names = url_features_df.columns.tolist() if url_features_df is not None else []
    return X_url, X_text, X_fused, y, url_feature_names, text_vec

def main(dataset_path: str = None, output_dir: str = "models"):
    if dataset_path is None:
        dataset_path = "data/raw/dataset.csv"
    df = load_dataset(dataset_path)
    url_col, text_col, label_col = _detect_columns(df)
    X_url, X_text, X_fused, y, url_feature_names, text_vec = _prepare_features_from_df(df, url_col, text_col, label_col)
    # 逐步划分训练/测试集
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    X_url_train, X_url_test = X_url[train_idx], X_url[test_idx]
    X_text_train, X_text_test = X_text[train_idx], X_text[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # 训练早期融合模型
    model_path = os.path.join(output_dir, "early_fusion.pkl")
    clf = train_early_fusion_from_slices(X_url_train, X_text_train, y_train, model_path=model_path)
    # 测试评估
    X_test_fused = hstack([X_url_test, X_text_test]) if X_text_test.shape[1] > 0 else X_url_test
    y_score = clf.predict_proba(X_test_fused)[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, y_score)
    # 简要阈值评估
    y_pred = (y_score >= 0.5).astype(int)
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # 保存评估结果
    metrics = {
        "auc": float(auc),
        "accuracy": float(acc),
        "f1": float(f1)
    }
    with open(os.path.join(output_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("训练完成. 评估指标:")
    print(metrics)
    print("模型已保存到:", model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phish detector 训练管线（早期融合）")
    parser.add_argument("--dataset", type=str, default=None, help="数据集路径，例如 data/raw/dataset.csv")
    parser.add_argument("--out", type=str, default="models", help="输出模型目录")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    main(dataset_path=args.dataset, output_dir=args.out)
