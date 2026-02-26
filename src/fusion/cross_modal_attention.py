import numpy as np
from scipy.sparse import csr_matrix

def cross_modal_attention(url_vec, text_vec):
    """
    Lightweight cross-modal interaction: average fused vector and a simple score.
    url_vec / text_vec may be dense or sparse; this is a placeholder to illustrate the idea.
    Returns fused vector and a dummy attention score.
    """
    if isinstance(url_vec, csr_matrix) and isinstance(text_vec, csr_matrix):
        fused = 0.5 * (url_vec + text_vec)
        score = float(min(1.0, (url_vec.power(2).sum() + text_vec.power(2).sum()) / max(1e-6, url_vec.power(2).sum() + text_vec.power(2).sum())))
        return fused, score
    # Fallback to numpy arrays
    nu = np.asarray(url_vec).reshape(-1)
    nt = np.asarray(text_vec).reshape(-1)
    fused = 0.5 * (nu + nt)
    score = float(np.linalg.norm(fused) / max(1.0, np.linalg.norm(nu) + np.linalg.norm(nt)))
    return fused, score
