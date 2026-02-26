# Contributions guide

- 环境准备
  - 确保已安装 Python 3.10+，并在虚拟环境中工作
- 初始化仓库
  - git init
  - git add -A
  - git commit -m "chore: initial project skeleton for phishing detector with LLM fusion"
- 工作流
  - 建立分支策略：main -> feat/** 进行新特性开发 -> PR 合并回 main
- 运行训练与演示
  - 数据集放在 data/raw/
  - 运行训练：python -m src.train_pipeline --dataset data/raw/your_dataset.csv --out models
  - 运行演示 API：uvicorn src.api.server:app --host 0.0.0.0 --port 8000
- 许可与数据
  - 遵循数据来源许可，避免使用受限数据；在论文/报告中注记数据来源与许可
