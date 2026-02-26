import os
from typing import Dict, Any

def llm_explain(label: int, features: Dict[str, Any], text: str = "") -> str:
    """
    Generate a simple explanation using an LLM if available.
    Falls back to a heuristic explanation when no API key is configured.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    try:
        import openai  # type: ignore
        if api_key:
            openai.api_key = api_key
            prompt = f"Explain the model decision. label={label}, features={features}, text='{text}'"
            resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
            return resp.choices[0].message.get("content", "No explanation from LLM.")
    except Exception:
        pass
    # Fallback explanation
    top = sorted(features.items(), key=lambda kv: float(kv[1] if isinstance(kv[1], (int, float)) else 0.0), reverse=True)[:5]
    parts = [f"{k}={v}" for k, v in top]
    return f"(fallback) Explanation based on top features: {', '.join(parts)}."
