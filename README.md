基于LLM与多模态线索融合的钓鱼检测系统

- 目标：高精度、可解释、具备抗对抗性与跨域泛化能力
- 设计：三种融合策略（early/late/cross-modal attention），LLM用于推理与解释
- 数据管线：抓取 → 清洗 → 特征抽取 → 模型训练 → 评估 → 演示
- 产出：训练脚本、Docker镜像、FastAPI演示接口、数据许可说明

结构
- data/: 原始数据与处理后的数据
- src/: 核心实现
- docker/: 容器化描述
- docs/: 数据许可、使用说明

快速开始
- Python 3.10+，建议使用虚拟环境
- pip install -r requirements.txt
- 数据与训练
  - 数据集请放在 data/raw/ 目录下，脚本会自动探测并加载第一份 CSV 文件
  - 训练脚本：python -m src.train_pipeline --dataset data/raw/your_dataset.csv --out models
- 运行演示：uvicorn src.api.server:app --reload --port 8000
- 数据来源：应遵循公开数据集许可，避免敏感数据
- 运行演示：uvicorn src.api.server:app --reload --port 8000
- 数据来源：应遵循公开数据集许可，避免敏感数据

后续跟进
- 根据阶段结果迭代融合策略与LLM提示
- 增加对抗性测试、跨域评估与可解释性评估
- 数据来源：应遵循公开数据集许可，避免敏感数据
- 训练脚本：python -m src.train_pipeline --dataset data/raw/your_dataset.csv --out models
- 数据来源：应遵循公开数据集许可，避免敏感数据
- 运行演示：uvicorn src.api.server:app --reload --port 8000
- 数据来源：应遵循公开数据集许可，避免敏感数据
- 数据可视化：执行 python scripts/visualize_all.py --outputs outputs
- 结果解读模板：docs/results_interpretation_template.md
