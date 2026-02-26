import numpy as np

def fuse_proba(url_proba, text_proba, method: str = "avg", weights=None) -> float:
    if method == "avg":
        return (float(url_proba) + float(text_proba)) / 2.0
    if method == "weights" and weights is not None:
        w_url, w_text = weights
        return w_url * url_proba + w_text * text_proba
    # default fallback
    return 0.5 * (float(url_proba) + float(text_proba))
