from setuptools import setup, find_packages

setup(
  name="phish_detector",
  version="0.1.0",
  description="LLM + multi-modal phishing detector",
  packages=find_packages(where="src"),
  package_dir={"": "src"},
  python_requires=">=3.10",
  install_requires=[
    "numpy>=1.21",
    "pandas>=1.3",
    "scikit-learn>=0.24",
    "fastapi>=0.95",
    "uvicorn>=0.22",
    "requests>=2.25",
    "python-dotenv>=1.0",
    "scipy>=1.8",
  ],
)
