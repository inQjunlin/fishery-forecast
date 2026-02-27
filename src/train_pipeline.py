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
from typing import Optional

from src.data_loader import load_dataset
from src.feature_extraction import extract_url_features, vectorize_texts
from src.fusion.early_fusion import train_early_fusion_from_slices
from src.pipeline.tabular_multimodal import train_baseline_numeric, train_tabular_multimodal, train_cross_modal_attention
from src.visualization import plot_roc_curve, plot_pr_curve
import json

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

def main(dataset_path: Optional[str] = None, output_dir: str = "models", mode: str = "baseline_numeric"):
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
    if mode == "baseline_numeric":
        # 仅基于数值特征的基线模型
        model_path, metrics, roc_data, pr_data = train_baseline_numeric(df, out_dir=output_dir, label_col='label', test_size=0.2, random_state=42)
        print("Baseline Numeric Metrics:", metrics)
        # 产出可视化曲线与数据
        mode_out = os.path.join(output_dir, mode)
        os.makedirs(mode_out, exist_ok=True)
        # 保存数据供后续可重复使用
        with open(os.path.join(mode_out, 'roc_data.json'), 'w', encoding='utf-8') as f:
            json.dump(roc_data, f, ensure_ascii=False, indent=2)
        with open(os.path.join(mode_out, 'pr_data.json'), 'w', encoding='utf-8') as f:
            json.dump(pr_data, f, ensure_ascii=False, indent=2)
        # 画图：确保 PR 数据包含 y_true/y_score，如果缺失则回填自适应信息
        if 'y_true' not in pr_data or 'y_score' not in pr_data:
            if roc_data is not None:
                pr_data['y_true'] = roc_data.get('y_true', [])
                pr_data['y_score'] = roc_data.get('y_score', [])
        if 'y_true' not in pr_data:
            pr_data['y_true'] = y_test.tolist() if isinstance(y_test, np.ndarray) else []
        if 'y_score' not in pr_data:
            pr_data['y_score'] = y_score.tolist() if isinstance(y_score, np.ndarray) else []
        # 绘制 ROC/PR 曲线
        roc_y_true = roc_data.get('y_true', [])
        roc_y_score = roc_data.get('y_score', [])
        # 先绘制 ROC 曲线
        plot_roc_curve(roc_y_true, roc_y_score, save_path=os.path.join(mode_out, 'roc_curve.png'), title='ROC - Baseline Numeric')
        # 安全地获取 PR 的 y_true/y_score，优先使用 pr_data 中的字段，其次回退到 ROC 数据
        pr_y_true = pr_data.get('y_true', roc_y_true)
        pr_y_score = pr_data.get('y_score', roc_y_score)
        try:
            plot_pr_curve(pr_y_true if pr_y_true is not None else roc_y_true,
                          pr_y_score if pr_y_score is not None else roc_y_score,
                          save_path=os.path.join(mode_out, 'pr_curve.png'), title='PR - Baseline Numeric')
        except Exception:
            # 回退至 ROC 数据的 PR 曲线作为兜底
            plot_pr_curve(roc_y_true, roc_y_score, save_path=os.path.join(mode_out, 'pr_curve.png'), title='PR - Baseline Numeric (fallback)')
        return
    elif mode == "multimodal":
        model_path, metrics, roc_data, pr_data = train_tabular_multimodal(df, out_dir=output_dir, label_col='label', test_size=0.2, random_state=42)
        print("Tabular Multimodal Metrics:", metrics)
        mode_out = os.path.join(output_dir, mode)
        os.makedirs(mode_out, exist_ok=True)
        with open(os.path.join(mode_out, 'roc_data.json'), 'w', encoding='utf-8') as f:
            json.dump(roc_data, f, ensure_ascii=False, indent=2)
        with open(os.path.join(mode_out, 'pr_data.json'), 'w', encoding='utf-8') as f:
            json.dump(pr_data, f, ensure_ascii=False, indent=2)
        roc_y_true = roc_data.get('y_true', [])
        roc_y_score = roc_data.get('y_score', [])
        plot_roc_curve(roc_y_true, roc_y_score, save_path=os.path.join(mode_out, 'roc_curve.png'), title='ROC - Tabular Multimodal')
        pr_y_true = pr_data.get('y_true', roc_y_true)
        pr_y_score = pr_data.get('y_score', roc_y_score)
        plot_pr_curve(pr_y_true, pr_y_score, save_path=os.path.join(mode_out, 'pr_curve.png'), title='PR - Tabular Multimodal')
        return
    elif mode == "crossmodal":
        model_path, metrics, roc_data, pr_data = train_cross_modal_attention(df, out_dir=output_dir, label_col='label', test_size=0.2, random_state=42)
        print("Cross-Modal Attention Metrics:", metrics)
        mode_out = os.path.join(output_dir, mode)
        os.makedirs(mode_out, exist_ok=True)
        with open(os.path.join(mode_out, 'roc_data.json'), 'w', encoding='utf-8') as f:
            json.dump(roc_data, f, ensure_ascii=False, indent=2)
        with open(os.path.join(mode_out, 'pr_data.json'), 'w', encoding='utf-8') as f:
            json.dump(pr_data, f, ensure_ascii=False, indent=2)
        roc_y_true = roc_data.get('y_true', [])
        roc_y_score = roc_data.get('y_score', [])
        plot_roc_curve(roc_y_true, roc_y_score, save_path=os.path.join(mode_out, 'roc_curve.png'), title='ROC - Cross-Modal Attention')
        pr_y_true = pr_data.get('y_true', roc_y_true)
        pr_y_score = pr_data.get('y_score', roc_y_score)
        plot_pr_curve(pr_y_true, pr_y_score, save_path=os.path.join(mode_out, 'pr_curve.png'), title='PR - Cross-Modal Attention')
        return
    else:
        # default: 早期融合骨架（兼容前面实现）
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
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("训练完成. 评估指标:")
    print(metrics)
    print("模型已保存到:", model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phish detector 训练管线（早期融合）")
    parser.add_argument("--dataset", type=str, default=None, help="数据集路径，例如 data/raw/dataset.csv")
    parser.add_argument("--out", type=str, default="models", help="输出模型目录")
    parser.add_argument("--mode", type=str, default="baseline_numeric", help="模式：baseline_numeric|multimodal|crossmodal")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    main(dataset_path=args.dataset, output_dir=args.out, mode=args.mode)
