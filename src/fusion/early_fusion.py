import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
import pickle

def train_early_fusion(X_url, X_text, y, model_path: str = "models/early_fusion.pkl") -> LogisticRegression:
    """
    Early fusion: concatenate URL features (dense) with text features (sparse),
    train a simple logistic regression classifier.
    """
    # Ensure URL features are in sparse form
    if not isinstance(X_text, csr_matrix):
        X_text = csr_matrix(X_text)
    if not isinstance(X_url, csr_matrix):
        X_url = csr_matrix(X_url)
    X_fused = hstack([X_url, X_text])
    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(X_fused, y)
    with open(model_path, "wb") as f:
        pickle.dump({"model": clf}, f)
    return clf

def predict_proba_early_fusion(clf, X_url, X_text) -> float:
    if not isinstance(X_text, csr_matrix):
        X_text = csr_matrix(X_text)
    if not isinstance(X_url, csr_matrix):
        X_url = csr_matrix(X_url)
    X_fused = hstack([X_url, X_text])
    return clf.predict_proba(X_fused)[:, 1]

def train_early_fusion_from_slices(X_url_train, X_text_train, y_train, model_path: str = "models/early_fusion.pkl"):
    """
    通过训练集分片来训练早期融合模型，返回训练好的分类器。
    输入为分片后的 URL 特征与文本特征（都应为 csr_matrix），以支持稀疏矩阵。
    """
    if not isinstance(X_url_train, csr_matrix):
        X_url_train = csr_matrix(X_url_train)
    if not isinstance(X_text_train, csr_matrix):
        X_text_train = csr_matrix(X_text_train)
    X_fused = hstack([X_url_train, X_text_train]) if X_text_train.shape[1] > 0 else X_url_train
    clf = LogisticRegression(max_iter=2000, n_jobs=-1, solver="saga")
    clf.fit(X_fused, y_train)
    # 保存模型对象，供后续加载推理
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    return clf
