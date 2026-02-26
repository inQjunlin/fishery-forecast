import re
from urllib.parse import urlparse
from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
import numpy as np

def extract_url_features(url: str) -> Dict[str, float]:
    """
    Extract a simple set of URL/structure features.
    """
    if not url:
        url = ""
    # Ensure scheme for parsing
    if not url.startswith("http"):
        url = "http://" + url
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path or ""
    query = parsed.query or ""
    url_len = len(url)
    has_https = int(url.startswith("https://"))
    has_ip = int(bool(re.search(r"(\\d{1,3}\\.){3}\\d{1,3}", domain)))
    subdomains = domain.split(".")
    n_subdomains = max(0, len(subdomains) - 2)
    n_queries = len(query.split("&")) if query else 0
    n_path_segments = len([p for p in path.split("/") if p])
    has_at_symbol = int("@" in url)
    contains_http = int("http://" in url or url.startswith("https://"))
    tld = domain.split(".")[-1] if domain else ""

    feats = {
        "url_len": float(url_len),
        "has_https": float(has_https),
        "has_ip": float(has_ip),
        "n_subdomains": float(n_subdomains),
        "n_queries": float(n_queries),
        "n_path_segments": float(n_path_segments),
        "has_at_symbol": float(has_at_symbol),
        "contains_http": float(contains_http),
        "tld_is_common": float(1.0 if tld in {"com", "org", "net", "edu"} else 0.0),
    }
    return feats

def vectorize_texts(texts, max_features: int = 5000):
    """
    Build and apply a TF-IDF vectorizer on a list of texts.
    Returns the sparse matrix and the fitted vectorizer.
    """
    vec = TfidfVectorizer(max_features=max_features, stop_words="english")
    X = vec.fit_transform([t for t in texts])
    return X, vec

def fuse_features(x_num: 'np.ndarray', x_text: 'csr_matrix'):
    """
    Stack numeric features (as dense) with text features (sparse).
    Returns a mixed sparse/dense matrix suitable for linear models.
    """
    if not isinstance(x_num, csr_matrix):
        x_num = csr_matrix(x_num)
    return hstack([x_num, x_text])
