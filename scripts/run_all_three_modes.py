#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可复现的流水线：依次训练三个方向（baseline_numeric、multimodal、crossmodal），
并自动生成 ROC/PR 曲线与特征重要性图，汇总到 outputs 目录。
"""
import os
import subprocess
import argparse

def run_mode(dataset_path: str, out_dir: str, mode: str):
    cmd = ["python", "-m", "src.train_pipeline", "--dataset", dataset_path, "--out", out_dir, "--mode", mode]
    print("Running:", " ".join(cmd))
    subprocess.run(" ".join(cmd), check=True, shell=True)

def main(dataset_path: str = "data/raw/dataset.csv", outputs_dir: str = "models"):
    modes = ["baseline_numeric", "multimodal", "crossmodal"]
    for mode in modes:
        run_mode(dataset_path, outputs_dir, mode)
    # 可视化汇总
    print("运行完毕，开始汇总可视化结果...")
    import subprocess
    subprocess.run(["python", "scripts/visualize_all.py", "--outputs", "outputs"], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="三路流水线一键执行示例")
    parser.add_argument("--dataset", default="data/raw/dataset.csv", help="数据集路径")
    parser.add_argument("--out", default="models", help="输出模型目录（分模态目录在其中）")
    args = parser.parse_args()
    main(dataset_path=args.dataset, outputs_dir=args.out)
