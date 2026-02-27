#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化汇总脚本：读取 outputs/{mode}/roc_data.json 和 pr_data.json，生成统一的 ROC/PR 曲线以及特征重要性图。
依赖：matplotlib
"""
import os
import json
import argparse
from pathlib import Path
import numpy as np

from src.visualization import plot_roc_curve, plot_pr_curve, plot_feature_importances

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main(outputs_root: str = 'models', top_n: int = 20):
    root = Path(outputs_root).parent if Path(outputs_root).name == 'models' else Path(outputs_root)
    # 假设 outputs 结构：outputs/{mode}/roc_data.json, pr_data.json, metrics.json
    modes = ['baseline_numeric', 'multimodal', 'crossmodal']
    for mode in modes:
        mode_dir = Path(outputs_root) / mode
        roc_path = mode_dir / 'roc_data.json'
        pr_path = mode_dir / 'pr_data.json'
        if not roc_path.exists():
            print(f"[WARN] 未找到 ROC 数据: {roc_path}")
            continue
        roc = load_json(roc_path)
        pr = load_json(pr_path) if pr_path.exists() else None
        # 画 ROC/PR 曲线
        plot_path_roc = mode_dir / 'roc_curve.png'
        plot_path_pr = mode_dir / 'pr_curve.png'
        plot_roc_curve(roc['y_true'], roc['y_score'], str(plot_path_roc), title=f"ROC - {mode}")
        if pr:
            plot_pr_curve(pr['y_true'], pr['y_score'], str(plot_path_pr), title=f"PR - {mode}")
        print(f"[INFO] 已生成曲线：{plot_path_roc}, {plot_path_pr if pr else '无 PR 曲线'}")

    # 特征重要性（仅在可获取特征名称与系数的模型上工作）
    # Baseline Numeric 示例
    base_model_path = Path(outputs_root) / 'baseline_numeric' / 'baseline_numeric_model.pkl'
    if base_model_path.exists():
        try:
            import joblib
            payload = joblib.load(base_model_path)
            coef = getattr(payload['model'], 'coef_', None)
            feature_names = payload.get('features', [])
            if coef is not None:
                weights = coef.flatten()
                viz_path = Path(outputs_root) / 'baseline_numeric' / 'feature_importances.png'
                plot_feature_importances(weights, feature_names, str(viz_path), top_n=top_n, title='Top Features - Baseline Numeric')
                print(f"[INFO] 已生成 Baseline Numeric 特征重要性图: {viz_path}")
        except Exception as e:
            print(f"[WARN] 无法生成 Baseline Numeric 特征重要性图: {e}")

    # Multimodal 示例：模型结构较复杂，试图获取 feature names
    mm_model_path = Path(outputs_root) / 'multimodal' / 'tabular_multimodal_model.pkl'
    if mm_model_path.exists():
        try:
            import joblib
            payload = joblib.load(mm_model_path)
            pipe = payload.get('model')
            numeric_features = payload.get('numeric_features', [])
            categorical_features = payload.get('categorical_features', [])
            # 尝试获取特征名
            if hasattr(pipe, 'named_steps') and 'preprocessor' in pipe.named_steps:
                pre = pipe.named_steps['preprocessor']
                if hasattr(pre, 'get_feature_names_out'):
                    feature_names = pre.get_feature_names_out(numeric_features + categorical_features).tolist()
                else:
                    feature_names = [str(n) for n in range(len(numeric_features))]
            else:
                feature_names = []
            coef = getattr(pipe, 'coef_', None)
            if coef is not None:
                weights = coef.flatten()
                viz_path = Path(outputs_root) / 'multimodal' / 'feature_importances.png'
                plot_feature_importances(weights, feature_names, str(viz_path), top_n=top_n, title='Top Features - Multimodal')
                print(f"[INFO] 已生成 Multimodal 特征重要性图: {viz_path}")
        except Exception as e:
            print(f"[WARN] 无法生成 Multimodal 特征重要性图: {e}")

    # Cross-modal Attention 我们将简单展示分支预测分布（如果数据可用）
    cross_model_path = Path(outputs_root) / 'crossmodal' / 'cross_modal_attention_model.pkl'
    if cross_model_path.exists():
        print(f"[INFO] Cross-Modal 曲线已在各自模式文件夹中绘制，如需更详细的解释，建议在训练时显式导出分支输出数据。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='可视化汇总：ROC/PR与特征重要性')
    parser.add_argument('--outputs', default='outputs', help='输出根目录')
    parser.add_argument('--top_n', type=int, default=20, help='特征重要性图显示的 TOPN')
    args = parser.parse_args()
    main(args.outputs, top_n=args.top_n)
