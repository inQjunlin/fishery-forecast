# -*- coding: utf-8 -*-
"""
数据加载工具模块
提供从 data/raw 目录中加载 CSV 数据集的简单接口
以及对数据列的自适应处理，方便后续训练管线接入。
"""

import os
import pandas as pd
from typing import Optional

def find_csv_in_dir(dir_path: str = "data/raw") -> str:
    """
    在指定目录下查找一个 CSV 文件并返回其完整路径
    如果目录下没有 CSV，将抛出异常
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"数据目录不存在: {dir_path}")
    for fname in os.listdir(dir_path):
        if fname.lower().endswith(".csv"):
            return os.path.join(dir_path, fname)
    raise FileNotFoundError(f"在目录 {dir_path} 下未找到 CSV 文件")

def load_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """
    从给定路径加载数据集；如果 path 未提供，则自动在 data/raw 目录查找一个 CSV。
    返回一个 pandas DataFrame。
    """
    if path is None:
        path = find_csv_in_dir("data/raw")
    return pd.read_csv(path)
