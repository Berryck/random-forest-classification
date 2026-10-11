# 🌳 随机森林分类算法完整教程 | Random Forest Classification Tutorial

<div align="center">


[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![WeChat](https://img.shields.io/badge/微信公众号-Python机器学习ML-brightgreen.svg)](#关注我们)

**一个系统化、工业级的随机森林分类算法教学项目**

涵盖从基础到高级的完整机器学习实战流程

[English](#english-version) | [中文](#中文版本)

</div>

---

## 📢 关注我们

<div align="center">


### 🎯 微信公众号：**Python机器学习ML**

### 📱 微信号：**zyfxw888**

<img src="https://img.shields.io/badge/扫码关注-获取更多机器学习教程-red?style=for-the-badge" alt="关注公众号"/>

**持续更新机器学习、深度学习、数据科学干货教程**

</div>

---

## 🌟 项目亮点

### 🎓 三级递进式教学体系

| 层级                  | 内容                                       | 适合人群               |
| --------------------- | ------------------------------------------ | ---------------------- |
| **初级 BASIC**        | 数据探索、划分、模型训练、正确评估         | 机器学习入门者         |
| **进阶 INTERMEDIATE** | 调参优化、特征重要性、PDP/ICE、学习曲线    | 有一定基础的学习者     |
| **高级 ADVANCED**     | 不平衡处理、SHAP解释、多模型对比、概率校准 | 追求工业级实践的开发者 |

### 📊 18张高质量可视化图表

- 混淆矩阵热力图
- 调参曲线（OOB/CV）
- 特征重要性三合一对比（不纯度/置换/SHAP）
- PDP 部分依赖图 + ICE 个体条件期望
- 学习曲线（样本量影响）
- 多模型 ROC 对比
- 概率校准曲线
- SHAP 摘要图与深度解析
- 评估仪表盘（六合一）
- 调参热力图
- 决策边界可视化
- 数据全景分析
- 特征诊断矩阵
- 多模型横评六合一
- 业务决策画布
- 误分类诊断

---

## 📁 项目结构

```
random-forest-classification/
├── 随机森林分类算法_完整版.py    # 主程序（初级/进阶/高级三层内容）
├── 数据13.1.csv                   # 信贷违约数据集（700条样本）
├── outputs_figures/               # 所有生成的可视化图表（18张）
│   ├── 01_基础_混淆矩阵热力图.png
│   ├── 02_进阶_调参曲线.png
│   ├── ...
│   └── 18_高级_误分类诊断.png
├── requirements.txt               # Python依赖包
├── README.md                      # 本文档
└── LICENSE                        # MIT许可证
```

---

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- 推荐使用 Anaconda 或 虚拟环境

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行程序

```bash
python 随机森林分类算法_完整版.py
```

### 4. 查看结果

所有生成的图表保存在 `outputs_figures/` 目录下

---

## 📖 核心内容

### 数据集介绍

- **来源**：信贷违约数据
- **样本量**：700 条
- **目标变量**：credit（1=违约，0=未违约）
- **特征变量**：
  - age：年龄
  - education：受教育程度
  - workyears：工作年限
  - resideyears：居住年限
  - income：年收入
  - debtratio：债务收入比
  - creditdebt：信用卡负债
  - otherdebt：其他负债

### 核心技术点

#### 🔰 初级内容（BASIC）

- 探索性数据分析（EDA）
- 分层采样划分训练/测试集
- 单决策树 vs 随机森林对比
- 袋外估计（OOB）原理
- 正确的模型评估方法
- 混淆矩阵可视化

#### 🔼 进阶内容（INTERMEDIATE）

- 使用 OOB 选择树的数量
- 交叉验证选择最优特征数
- GridSearchCV 联合调参
- 置换重要性（Permutation Importance）
- PDP 部分依赖图
- ICE 个体条件期望
- 学习曲线分析

#### 🔥 高级内容（ADVANCED）

- 类别不平衡处理（class_weight）
- 分类阈值优化
- 多模型横向对比（Logistic/Tree/Bagging/GBDT/HistGBDT）
- 概率校准曲线
- SHAP 模型解释
- 代价曲线、决策曲线、KS曲线、Lift曲线
- 综合评估仪表盘

---

## 🎨 可视化示例

### 特征重要性三合一对比

对比不纯度法、置换重要性、SHAP三种方法，避免单一指标的偏差

### SHAP 深度解析

深入理解每个特征对模型预测的实际贡献

### 多模型横评六合一

一张图对比6种模型在多个指标上的表现

### 评估仪表盘

ROC、PR曲线、校准曲线、混淆矩阵、阈值权衡、概率分布六合一

---

## 📚 学习路径建议

1. **初学者**：
   - 先运行完整代码，观察所有输出
   - 阅读 `改进说明文档.md` 理解常见错误
   - 逐段理解初级（BASIC）部分代码

2. **有基础的学习者**：
   - 重点学习进阶（INTERMEDIATE）部分的调参技巧
   - 理解不同特征重要性方法的差异
   - 掌握 PDP/ICE 的业务解释

3. **追求工业实践**：
   - 深入研究高级（ADVANCED）部分
   - 学习 SHAP 模型解释
   - 掌握多模型对比与选择方法
   - 理解概率校准的重要性

---

## 🔧 技术栈

- **核心框架**：scikit-learn
- **数据处理**：pandas, numpy
- **可视化**：matplotlib, seaborn
- **模型解释**：shap
- **可选**：mlxtend（决策边界绘制）

---

## 📝 常见问题

### Q1: 为什么不用测试集选参数？

**A**: 这会导致数据泄露，测试集不再是"未见过的数据"，最终评估会偏乐观。应使用 OOB 或交叉验证在训练集内部选参。

### Q2: 为什么不只看准确率？

**A**: 在类别不平衡场景下，准确率具有误导性。需要结合 Precision、Recall、F1、AUC、Kappa 等多指标综合评估。

### Q3: 三种特征重要性方法有什么区别？

**A**: 

- **不纯度法**：快速但对连续变量和高基数变量有偏
- **置换重要性**：在测试集上衡量泛化贡献，更可靠
- **SHAP**：基于博弈论，提供最全面的解释

### Q4: SHAP 计算很慢怎么办？

**A**: 可以使用 TreeExplainer（针对树模型优化）或采样部分样本计算

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

如果这个项目对你有帮助，请：

1. ⭐ Star 本项目
2. 📢 分享给更多学习者
3. 🔔 关注微信公众号 **Python机器学习ML** 获取更多内容

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议

---

## 🙏 致谢

- 感谢 scikit-learn 团队提供优秀的机器学习工具
- 感谢 SHAP 团队提供强大的模型解释工具
- 感谢所有为开源社区做出贡献的开发者

---

<div align="center">


## 💬 联系方式

| 平台         | 信息                                                     |
| ------------ | -------------------------------------------------------- |
| 📱 微信公众号 | **Python机器学习ML**                                     |
| 🆔 微信号     | **zyfxw888**                                             |
| 🌐 GitHub     | [https://github.com/Berryck](https://github.com/Berryck) |

### ⬇️ 扫码关注获取更多教程 ⬇️
<img width="279" height="286" alt="image" src="https://github.com/user-attachments/assets/7a7ba060-eeed-49df-87f4-a6bc59378962" />

**持续更新机器学习、数据挖掘、Python编程干货**

---

**如果这个项目帮助到了你，请点个 ⭐ Star！**

---

Made with ❤️ by Python机器学习ML团队

</div>

---

## English Version

# 🌳 Random Forest Classification - Complete Tutorial

A comprehensive, production-level Random Forest classification tutorial covering beginner to advanced machine learning workflows.

## 🌟 Key Features

- ✅ Fixed critical bugs from original code (savefig, data leakage, evaluation methods)
- 📚 Three-tier progressive learning system (Basic → Intermediate → Advanced)
- 📊 18 high-quality visualization charts
- 🔍 Comprehensive model interpretation (SHAP, PDP, ICE)
- ⚖️ Imbalanced classification handling
- 🎯 Multi-model comparison framework

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the script
python 随机森林分类算法_完整版.py

# Check outputs in outputs_figures/ directory
```

## 📖 What You'll Learn

### Basic Level

- Exploratory Data Analysis
- Proper train/test splitting
- Model training and evaluation
- Confusion matrix visualization

### Intermediate Level

- Hyperparameter tuning with OOB/CV
- Feature importance analysis
- Partial Dependence Plots (PDP)
- Individual Conditional Expectation (ICE)
- Learning curves

### Advanced Level

- Imbalanced class handling
- Threshold optimization
- Multi-model comparison
- Probability calibration
- SHAP model interpretation
- Comprehensive evaluation dashboards

## 📊 Dataset

**Credit Default Dataset**

- 700 samples
- Binary classification (default/non-default)
- 8 features: age, education, work years, residence years, income, debt ratio, credit card debt, other debt

## 🎨 Visualizations

The script generates 18 professional-quality visualizations covering:

- Confusion matrices
- ROC and PR curves
- Feature importance comparisons
- SHAP analysis
- Model comparison dashboards
- Decision boundaries
- Calibration curves
- And more...

## 🤝 Contributing

Issues and Pull Requests are welcome!

If this project helps you:

1. ⭐ Star this repository
2. 📢 Share with other learners
3. 🔔 Follow our WeChat Official Account: **Python机器学习ML**

## 📱 Follow Us

**WeChat Official Account**: Python机器学习ML  
**WeChat ID**: zyfxw888  
**GitHub**: [https://github.com/Berryck](https://github.com/Berryck)

Get continuous updates on machine learning, data science, and Python programming!

---

## 📄 License

MIT License

---

<div align="center">


**If this project helps you, please give it a ⭐ Star!**

Made with ❤️ for the Machine Learning Community

</div>
