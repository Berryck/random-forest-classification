# -*- coding: utf-8 -*-
"""
================================================================================
随机森林分类算法 —— 完整教学与实战脚本（初级 / 进阶 / 高级）
================================================================================
数据集       : 数据13.1.csv （信贷违约数据，700 条样本）
响应变量 y   : credit  （1=违约 / 0=未违约，二分类）
特征变量 X   : age 年龄, education 受教育程度, workyears 工作年限,
               resideyears 居住年限, income 年收入, debtratio 债务收入比,
               creditdebt 信用卡负债, otherdebt 其他负债

本脚本在原公众号示例基础上做了系统性完善，分三个层级递进：
  【初级 BASIC】        正确的建模基础流程（修复原文 savefig / 评价等 bug）
  【进阶 INTERMEDIATE】 无数据泄露的调参、置换重要性、PDP/ICE、学习曲线
  【高级 ADVANCED】     不平衡处理、概率校准、SHAP、模型横向对比、组合大图

原文主要问题与本脚本的修复（详见各节 "改进说明"）：
  1. savefig 写在 show 之后 → 保存空白图。本脚本统一在 show 前保存。
  2. 用测试集挑 max_features / n_estimators → 数据泄露。改用 OOB / 交叉验证。
  3. 决策边界用全量数据 fit 并自评 → 训练集自评虚高。改为训练集拟合网格可视化。
  4. 只看 accuracy → 不平衡场景失真。补 Recall/F1/AUC/PR/Kappa/阈值优化。
  5. 重要性只用不纯度法（对连续/高基数变量有偏）→ 补置换重要性 + SHAP。

运行方式：
    python 随机森林分类算法_完整版.py
所有图片输出到脚本同目录下的  outputs_figures/  文件夹。

依赖：numpy pandas matplotlib seaborn scikit-learn shap
      （可选 mlxtend；缺失时脚本会自动跳过对应绘图，不影响主流程）
================================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# 全局绘图配置：解决中文与负号显示；统一风格
# ----------------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC",
                                   "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False          # 正常显示负号
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.alpha"] = 0.3

RANDOM_STATE = 10                                   # 全局随机种子，保证可复现
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs_figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# 统一配色方案（取自指定的"六色"色卡，全脚本共用，保证视觉一致）
# ----------------------------------------------------------------------------
from matplotlib.colors import LinearSegmentedColormap
# 组1（主色系，用于大多数分类/序列着色）
C_TEAL   = "#009391"   # 主色·青绿   RGB(0,147,145)
C_ORANGE = "#EFB47A"   # 橙          RGB(239,180,122)
C_PURPLE = "#B3A9ED"   # 紫          RGB(179,169,237)
C_BLUEL  = "#C4E3F2"   # 浅蓝        RGB(196,227,242)
C_PINK   = "#F1BFDA"   # 粉          RGB(241,191,218)
C_GRAY   = "#DBDBDB"   # 浅灰        RGB(219,219,219)
# 组2（强调色系，用于对比/强调/第二组序列）
C_DARK   = "#525252"   # 深灰(文字/参考线) RGB(82,82,82)
C_RED    = "#F6847A"   # 红          RGB(246,132,122)
C_ORANGE2 = "#F8B583"  # 橙2         RGB(248,181,131)
C_BLUE   = "#4194C6"   # 蓝          RGB(65,148,198)
C_MAUVE  = "#CA87A7"   # 紫红        RGB(202,135,167)
C_BLUE2  = "#80A0CB"   # 蓝紫        RGB(128,160,203)

# 定性调色板（分类/多序列按顺序取色）
PALETTE = [C_TEAL, C_ORANGE, C_PURPLE, C_BLUE, C_PINK, C_RED,
           C_ORANGE2, C_MAUVE, C_BLUE2, C_BLUEL]
plt.rcParams["axes.prop_cycle"] = mpl.cycler(color=PALETTE)

# 二分类专用：未违约(负类,青绿) / 违约(正类,红)
C_NEG, C_POS = C_TEAL, C_RED

# 连续型渐变色带（由色卡主色自定义生成）
# 违约概率/诊断：低(青绿,安全) → 中(白) → 高(红,风险)
CMAP_RISK = LinearSegmentedColormap.from_list("risk", [C_TEAL, "#FFFFFF", C_RED])
CMAP_DIV = CMAP_RISK
# 通用顺序色：白 → 浅蓝 → 青绿（用于热力图/概率填充）
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_teal", ["#FFFFFF", C_BLUEL, C_TEAL])
# 蓝色系（用于混淆矩阵等）
CMAP_BLUE = LinearSegmentedColormap.from_list(
    "blues6", ["#FFFFFF", C_BLUEL, C_BLUE2, C_BLUE])


def savefig(name):
    """统一保存函数。

    关键修复：原文所有 plt.savefig() 都写在 plt.show() 之后，而 show() 会清空
    当前 figure，导致保存的是空白图。这里在 show 之前保存，集中管理输出路径。
    """
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path)
    print(f"  [图已保存] {path}")


def finalize(name):
    """保存 → 显示 → 关闭，一步完成（新增组合图统一使用）。"""
    savefig(name)
    plt.show()
    plt.close("all")


def section(title):
    """打印分节标题，让控制台输出层次清晰。"""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def precision_score_safe(y_true, y_pred):
    """精确率；当预测无正例时返回 0，避免除零告警。"""
    from sklearn.metrics import precision_score
    return precision_score(y_true, y_pred, zero_division=0)


def recall_score_safe(y_true, y_pred):
    """召回率；当无正例时返回 0，避免除零告警。"""
    from sklearn.metrics import recall_score
    return recall_score(y_true, y_pred, zero_division=0)


# ----------------------------------------------------------------------------
# sklearn 及相关库导入（集中管理，缺失的可选依赖自动降级）
# ----------------------------------------------------------------------------
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV,
    cross_val_score, learning_curve, validation_curve, cross_val_predict,
)
from sklearn.ensemble import (
    RandomForestClassifier, BaggingClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import (
    PartialDependenceDisplay, permutation_importance, partial_dependence,
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix, classification_report, cohen_kappa_score,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
    f1_score, accuracy_score,
)

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False
    print("[提示] 未检测到 shap 库，高级部分的 SHAP 解释将被跳过（pip install shap）。")

try:
    from mlxtend.plotting import plot_decision_regions
    HAS_MLXTEND = True
except Exception:
    HAS_MLXTEND = False

try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False

FEATURE_CN = {
    "age": "年龄", "education": "受教育程度", "workyears": "工作年限",
    "resideyears": "居住年限", "income": "年收入", "debtratio": "债务收入比",
    "creditdebt": "信用卡负债", "otherdebt": "其他负债",
}


# ----------------------------------------------------------------------------
# 数据加载（三个层级共用）
# ----------------------------------------------------------------------------
def load_data():
    """读取 数据13.1.csv，返回 data, X, y。"""
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "数据13.1.csv"
    )
    data = pd.read_csv(csv_path)
    X = data.iloc[:, 1:]      # 第 1 列之后为特征变量
    y = data.iloc[:, 0]       # 第 0 列 credit 为响应变量
    return data, X, y


# ============================================================================
#                           初级 BASIC
# ----------------------------------------------------------------------------
# 【什么时候用随机森林】
#   - 结构化 / 表格数据的分类或回归，特征间存在非线性、交互关系；
#   - 想要一个"开箱即用、几乎不用调参也能打"的强基线模型；
#   - 需要特征重要性做初步解释；样本量从几百到几百万都适用。
# 【怎么用】
#   1) 划分训练/测试集（分层抽样保持类别比例）；
#   2) 先跑一棵决策树看基线，再用随机森林对比，体会"集成"的收益；
#   3) 用混淆矩阵 + classification_report + Kappa 做正确评价（不要只看准确率）。
# 【注意事项】
#   - 随机森林对特征缩放不敏感，无需标准化（这点比逻辑回归/KNN 省心）；
#   - 不平衡数据不要只看 accuracy，务必看每一类的 recall/precision；
#   - 设定 random_state 保证结果可复现。
# ============================================================================
def run_basic():
    section("初级 BASIC —— 正确的随机森林建模基础流程")
    data, X, y = load_data()

    # ---- EDA：先认识数据，尤其是类别是否平衡 ----
    print("样本量: %d,  特征数: %d" % X.shape)
    print("缺失值总数:", int(data.isnull().sum().sum()))
    vc = y.value_counts().sort_index()
    ratio = vc / vc.sum()
    print("类别分布(0=未违约,1=违约):")
    for k in vc.index:
        print("    类别 %d: %d 条  (%.1f%%)" % (k, vc[k], ratio[k] * 100))
    print("→ 违约样本占比约 %.0f%%，属轻度不平衡，评价需关注违约类召回率。" %
          (ratio.get(1, 0) * 100))

    # ---- 分层划分训练/测试集 ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)

    # ---- 基线1：单棵决策树 ----
    tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)
    acc_tree = tree.score(X_test, y_test)
    print("\n[基线] 单棵决策树 测试集准确率: %.4f" % acc_tree)

    # ---- 随机森林 ----
    rf = RandomForestClassifier(
        n_estimators=300, max_features="sqrt",
        oob_score=True, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    acc_rf = rf.score(X_test, y_test)
    print("[随机森林] 测试集准确率: %.4f   OOB 袋外准确率: %.4f" %
          (acc_rf, rf.oob_score_))
    print("→ 森林相比单树提升 %.4f，体现了集成'群策群力'的收益。" %
          (acc_rf - acc_tree))

    # ---- 正确的性能评价 ----
    pred = rf.predict(X_test)
    print("\n分类报告:\n", classification_report(y_test, pred, digits=3))
    print("Cohen's Kappa: %.4f" % cohen_kappa_score(y_test, pred))

    # ---- 混淆矩阵热力图（修复：show 前保存）----
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5, 4))
    if HAS_SNS:
        sns.heatmap(cm, annot=True, fmt="d", cmap=CMAP_BLUE,
                    xticklabels=["未违约", "违约"], yticklabels=["未违约", "违约"])
    else:
        plt.imshow(cm, cmap=CMAP_BLUE)
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("预测类别"); plt.ylabel("真实类别")
    plt.title("初级-混淆矩阵热力图")
    savefig("01_基础_混淆矩阵热力图.png")
    plt.show(); plt.close()

    return dict(data=data, X=X, y=y, X_train=X_train, X_test=X_test,
                y_train=y_train, y_test=y_test)


# ============================================================================
#                           进阶 INTERMEDIATE
# ----------------------------------------------------------------------------
# 【什么时候用这些进阶手段】
#   - 当你要"认真"上线一个模型，而不只是跑个 demo 时；
#   - 需要在不泄露测试集的前提下挑参数、并解释模型行为时。
# 【怎么用】
#   1) 用 OOB 曲线选 n_estimators（森林自带的免费验证，无需额外划分）；
#   2) 用交叉验证 / GridSearchCV 选 max_features、max_depth 等（禁止用测试集）；
#   3) 用置换重要性纠正不纯度重要性对连续/高基数变量的偏好；
#   4) 用 PDP/ICE 看特征如何影响预测；用学习曲线判断欠拟合/过拟合。
# 【注意事项】
#   - 关键原则：任何"看着测试集调参"的做法都是数据泄露，评估会偏乐观；
#   - 置换重要性应在测试集上算（衡量泛化贡献），不纯度重要性来自训练；
#   - GridSearchCV 用分层 K 折，scoring 选 roc_auc 比 accuracy 更稳健。
# ============================================================================
def run_intermediate(ctx):
    section("进阶 INTERMEDIATE —— 无泄露调参、置换重要性、PDP/ICE、学习曲线")
    X, y = ctx["X"], ctx["y"]
    X_train, X_test = ctx["X_train"], ctx["X_test"]
    y_train, y_test = ctx["y_train"], ctx["y_test"]

    # ---- (1) OOB 曲线选 n_estimators ----
    n_range = list(range(50, 401, 25))
    oob_scores = []
    for n in n_range:
        m = RandomForestClassifier(n_estimators=n, max_features="sqrt",
                                   oob_score=True, random_state=RANDOM_STATE,
                                   n_jobs=-1)
        m.fit(X_train, y_train)
        oob_scores.append(m.oob_score_)
    best_n = n_range[int(np.argmax(oob_scores))]
    print("OOB 最优 n_estimators = %d (OOB=%.4f)" % (best_n, max(oob_scores)))

    # ---- (2) 交叉验证选 max_features（用训练集，不碰测试集）----
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    feat_range = range(1, X.shape[1] + 1)
    cv_means = []
    for mf in feat_range:
        m = RandomForestClassifier(n_estimators=best_n, max_features=mf,
                                   random_state=RANDOM_STATE, n_jobs=-1)
        s = cross_val_score(m, X_train, y_train, cv=cv, scoring="roc_auc")
        cv_means.append(s.mean())
    best_mf = list(feat_range)[int(np.argmax(cv_means))]
    print("交叉验证最优 max_features = %d (AUC=%.4f)" % (best_mf, max(cv_means)))

    # ---- 保存调参曲线组合图（OOB 与 CV 并排）----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(n_range, oob_scores, "o-", color=C_TEAL)
    axes[0].axvline(best_n, ls="--", color=C_DARK)
    axes[0].set_xlabel("n_estimators"); axes[0].set_ylabel("OOB 准确率")
    axes[0].set_title("(a) OOB 曲线选树的数量")
    axes[1].plot(list(feat_range), cv_means, "s-", color=C_RED)
    axes[1].axvline(best_mf, ls="--", color=C_DARK)
    axes[1].set_xlabel("max_features"); axes[1].set_ylabel("10折CV AUC")
    axes[1].set_title("(b) 交叉验证选最大特征数")
    plt.suptitle("进阶-无数据泄露的调参过程", fontweight="bold")
    savefig("02_进阶_调参曲线.png")
    plt.show(); plt.close()

    # ---- (3) GridSearchCV 联合调参 ----
    param_grid = {
        "max_depth": [None, 6, 10, 16],
        "min_samples_leaf": [1, 3, 5],
    }
    base = RandomForestClassifier(n_estimators=best_n, max_features=best_mf,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    grid = GridSearchCV(base, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    print("GridSearch 最优参数:", grid.best_params_)
    print("GridSearch 最优 CV AUC: %.4f" % grid.best_score_)
    best_rf = grid.best_estimator_
    ctx["best_rf"] = best_rf
    ctx["best_params"] = dict(n_estimators=best_n, max_features=best_mf,
                              **grid.best_params_)

    # ---- (4) 不纯度重要性 vs 置换重要性 ----
    imp_gini = best_rf.feature_importances_
    perm = permutation_importance(best_rf, X_test, y_test, n_repeats=20,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    imp_perm = perm.importances_mean
    order = np.argsort(imp_gini)
    labels = [FEATURE_CN.get(c, c) for c in X.columns]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    yy = np.arange(len(order))
    axes[0].barh(yy, imp_gini[order], color=C_TEAL)
    axes[0].set_yticks(yy); axes[0].set_yticklabels([labels[i] for i in order])
    axes[0].set_title("(a) 不纯度重要性(训练,对连续变量有偏)")
    axes[1].barh(yy, imp_perm[order], xerr=perm.importances_std[order],
                 color=C_BLUE)
    axes[1].set_title("(b) 置换重要性(测试,衡量泛化贡献)")
    plt.suptitle("进阶-两种特征重要性对比", fontweight="bold")
    savefig("03_进阶_重要性对比.png")
    plt.show(); plt.close()

    # ---- (5) PDP + ICE ----
    fig, ax = plt.subplots(figsize=(12, 4.5))
    PartialDependenceDisplay.from_estimator(
        best_rf, X_train, ["workyears", "debtratio", "income"],
        kind="both", ax=ax, n_jobs=-1)
    plt.suptitle("进阶-部分依赖图(PDP)与个体条件期望(ICE)", fontweight="bold")
    savefig("04_进阶_PDP_ICE.png")
    plt.show(); plt.close()

    # ---- (6) 学习曲线 ----
    train_sizes, tr, va = learning_curve(
        best_rf, X_train, y_train, cv=cv, scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(train_sizes, tr.mean(1), "o-", label="训练集")
    plt.plot(train_sizes, va.mean(1), "s-", label="交叉验证")
    plt.fill_between(train_sizes, va.mean(1)-va.std(1), va.mean(1)+va.std(1),
                     alpha=0.15)
    plt.xlabel("训练样本数"); plt.ylabel("AUC"); plt.legend()
    plt.title("进阶-学习曲线(判断欠/过拟合与增数据收益)")
    savefig("05_进阶_学习曲线.png")
    plt.show(); plt.close()
    return ctx


# ============================================================================
#                           高级 ADVANCED
# ----------------------------------------------------------------------------
# 【什么时候用这些高级手段】
#   - 业务对"漏判违约"代价敏感（风控、医疗、欺诈）→ 必须做不平衡处理与阈值优化；
#   - 模型输出的概率要直接用于决策/定价 → 必须做概率校准；
#   - 需要向业务方/监管解释"为什么预测这个人违约" → 用 SHAP 做个体级解释；
#   - 要论证选型合理性 → 做多模型横向对比。
# 【怎么用】
#   1) class_weight='balanced' 让森林关注少数类；用 PR 曲线而非只看 ROC；
#   2) 在验证集上按 F1 最大 / 业务代价最小选择分类阈值，而非默认 0.5；
#   3) 用 calibration_curve 看概率是否可信，必要时做 Platt/Isotonic 校准；
#   4) 用 shap 解释全局与个体；用统一接口对比 RF/GBDT/HistGBDT/Logistic。
# 【注意事项】
#   - 阈值必须在训练/验证集上选，不能在测试集上选（否则又是数据泄露）；
#   - 校准曲线要在独立数据上评估；SHAP 计算量大，可对样本抽样。
# ============================================================================
def run_advanced(ctx):
    section("高级 ADVANCED —— 不平衡处理、阈值优化、校准、SHAP、模型对比")
    X, y = ctx["X"], ctx["y"]
    X_train, X_test = ctx["X_train"], ctx["X_test"]
    y_train, y_test = ctx["y_train"], ctx["y_test"]
    params = ctx.get("best_params", dict(n_estimators=300, max_features="sqrt"))

    # ---- (1) 不平衡处理：class_weight='balanced' ----
    rf_bal = RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        **{k: v for k, v in params.items() if k in
           ("n_estimators", "max_features", "max_depth", "min_samples_leaf")})
    rf_bal.fit(X_train, y_train)
    prob_test = rf_bal.predict_proba(X_test)[:, 1]
    ctx["rf_bal"] = rf_bal
    ctx["prob_test"] = prob_test

    # ---- (2) 阈值优化：在训练集上按 F1 最大选阈值 ----
    prob_train = rf_bal.predict_proba(X_train)[:, 1]
    ths = np.linspace(0.1, 0.9, 81)
    f1s_tr = [f1_score(y_train, (prob_train >= t).astype(int)) for t in ths]
    best_t = ths[int(np.argmax(f1s_tr))]
    ctx["best_threshold"] = best_t
    pred_default = (prob_test >= 0.5).astype(int)
    pred_tuned = (prob_test >= best_t).astype(int)
    print("默认阈值0.5   F1(测试)=%.4f" % f1_score(y_test, pred_default))
    print("最优阈值%.3f F1(测试)=%.4f" % (best_t, f1_score(y_test, pred_tuned)))

    # ---- (3) 多模型横向对比 ----
    models = {
        "Logistic": LogisticRegression(max_iter=1000),
        "决策树": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "装袋法": BaggingClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                    n_jobs=-1),
        "随机森林": rf_bal,
        "GBDT": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "HistGBDT": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    rows = []
    roc_data = {}
    for name, mdl in models.items():
        if name != "随机森林":
            mdl.fit(X_train, y_train)
        p = mdl.predict_proba(X_test)[:, 1]
        pr = mdl.predict(X_test)
        auc = roc_auc_score(y_test, p)
        ap = average_precision_score(y_test, p)
        rows.append(dict(模型=name, 准确率=accuracy_score(y_test, pr),
                         AUC=auc, AP=ap, F1=f1_score(y_test, pr),
                         Kappa=cohen_kappa_score(y_test, pr)))
        roc_data[name] = (roc_curve(y_test, p), auc)
    cmp_df = pd.DataFrame(rows).set_index("模型").round(4)
    print("\n模型横向对比:\n", cmp_df)
    ctx["cmp_df"] = cmp_df
    ctx["roc_data"] = roc_data
    ctx["models"] = models

    # ---- 多模型 ROC 对比图 ----
    plt.figure(figsize=(6.5, 5.5))
    for name, ((fpr, tpr, _), auc) in roc_data.items():
        plt.plot(fpr, tpr, label="%s (AUC=%.3f)" % (name, auc))
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("假正率 FPR"); plt.ylabel("真正率 TPR")
    plt.title("高级-多模型 ROC 曲线对比"); plt.legend(fontsize=8)
    savefig("06_高级_多模型ROC对比.png")
    plt.show(); plt.close()

    # ---- (4) 概率校准曲线（可靠性图）----
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="完美校准")
    for name in ["随机森林", "GBDT", "Logistic"]:
        p = models[name].predict_proba(X_test)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10,
                                                 strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", label=name)
    ax.set_xlabel("预测概率"); ax.set_ylabel("实际违约比例")
    ax.set_title("高级-概率校准曲线(可靠性图)"); ax.legend()
    savefig("07_高级_概率校准曲线.png")
    plt.show(); plt.close()

    # ---- (5) SHAP 全局与个体解释 ----
    if HAS_SHAP:
        try:
            bg = X_train.sample(min(200, len(X_train)), random_state=RANDOM_STATE)
            Xs = X_test.sample(min(150, len(X_test)), random_state=RANDOM_STATE)
            explainer = shap.TreeExplainer(rf_bal)
            sv = explainer.shap_values(Xs)
            # 二分类取正类
            sv_pos = sv[1] if isinstance(sv, list) else (
                sv[..., 1] if sv.ndim == 3 else sv)
            ctx["shap_values"] = sv_pos
            ctx["shap_X"] = Xs
            # 基准值(正类)，供后续 waterfall 个体解释复用，避免重复计算
            ev = np.atleast_1d(explainer.expected_value)
            ctx["shap_base"] = float(ev[1] if ev.size > 1 else ev[0])
            plt.figure()
            shap.summary_plot(sv_pos, Xs, show=False,
                              feature_names=[FEATURE_CN.get(c, c)
                                             for c in Xs.columns])
            plt.title("高级-SHAP 摘要图(重要性+方向+分布 一图看全)")
            savefig("08_高级_SHAP摘要图.png")
            plt.show(); plt.close()
        except Exception as e:
            print("[SHAP] 计算失败，跳过：", e)
    else:
        print("[SHAP] 未安装 shap，跳过 SHAP 解释图。")

    return ctx


# ============================================================================
#   高级组合大图 A —— 模型评估仪表盘（2×3 子图，一张图看全性能）
# ----------------------------------------------------------------------------
# 设计思路：把评估一个分类器最常看的 6 张图合成一张，横纵坐标互相呼应，
#   便于一眼对比不同视角下的同一个模型：
#   (1) ROC 曲线    (2) PR 曲线    (3) 概率校准曲线
#   (4) 混淆矩阵    (5) 阈值-指标曲线(P/R/F1)   (6) 预测概率分布(按真实类)
# ============================================================================
def make_dashboard(ctx):
    section("高级组合图 A —— 模型评估仪表盘（2×3）")
    y_test = ctx["y_test"]
    prob = ctx["prob_test"]
    best_t = ctx.get("best_threshold", 0.5)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # (1) ROC
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax = axes[0, 0]
    ax.plot(fpr, tpr, color=C_RED, lw=2, label="AUC=%.3f" % auc)
    ax.fill_between(fpr, tpr, alpha=0.12, color=C_RED)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("假正率 FPR"); ax.set_ylabel("真正率 TPR")
    ax.set_title("(1) ROC 曲线"); ax.legend()

    # (2) PR
    prec, rec, _ = precision_recall_curve(y_test, prob)
    ap = average_precision_score(y_test, prob)
    ax = axes[0, 1]
    ax.plot(rec, prec, color=C_BLUE, lw=2, label="AP=%.3f" % ap)
    base = y_test.mean()
    ax.axhline(base, ls="--", color=C_DARK, label="基线=%.2f" % base)
    ax.set_xlabel("召回率 Recall"); ax.set_ylabel("精确率 Precision")
    ax.set_title("(2) PR 曲线(不平衡更敏感)"); ax.legend()

    # (3) 校准曲线
    frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10)
    ax = axes[0, 2]
    ax.plot(mean_pred, frac_pos, "o-", color=C_PURPLE, label="模型")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="完美校准")
    ax.set_xlabel("预测概率均值"); ax.set_ylabel("实际正例比例")
    ax.set_title("(3) 概率校准曲线"); ax.legend()

    # (4) 混淆矩阵(用最优阈值)
    pred = (prob >= best_t).astype(int)
    cm = confusion_matrix(y_test, pred)
    ax = axes[1, 0]
    im = ax.imshow(cm, cmap=CMAP_BLUE)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=13)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["未违约", "违约"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["未违约", "违约"])
    ax.set_xlabel("预测"); ax.set_ylabel("真实")
    ax.set_title("(4) 混淆矩阵(阈值=%.2f)" % best_t)

    # (5) 阈值-指标曲线
    ths = np.linspace(0.05, 0.95, 91)
    P, R, F = [], [], []
    for t in ths:
        pd_ = (prob >= t).astype(int)
        P.append(precision_score_safe(y_test, pd_))
        R.append(recall_score_safe(y_test, pd_))
        F.append(f1_score(y_test, pd_))
    ax = axes[1, 1]
    ax.plot(ths, P, label="Precision")
    ax.plot(ths, R, label="Recall")
    ax.plot(ths, F, label="F1", lw=2)
    ax.axvline(best_t, ls="--", color=C_DARK, label="最优阈值=%.2f" % best_t)
    ax.set_xlabel("分类阈值"); ax.set_ylabel("指标值")
    ax.set_title("(5) 阈值-指标权衡"); ax.legend(fontsize=8)

    # (6) 概率分布
    ax = axes[1, 2]
    ax.hist(prob[y_test == 0], bins=25, alpha=0.6, label="真实未违约", color=C_TEAL)
    ax.hist(prob[y_test == 1], bins=25, alpha=0.6, label="真实违约", color=C_POS)
    ax.axvline(best_t, ls="--", color=C_DARK)
    ax.set_xlabel("预测为违约的概率"); ax.set_ylabel("样本数")
    ax.set_title("(6) 预测概率分布(按真实类)"); ax.legend(fontsize=8)

    plt.suptitle("高级组合图 A · 随机森林模型评估仪表盘", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    savefig("09_高级_评估仪表盘.png")
    plt.show(); plt.close()


# ============================================================================
#   高级组合大图 B —— 特征重要性三合一（不纯度 + 置换 + SHAP 对齐同一坐标）
# ----------------------------------------------------------------------------
# 设计思路：三种口径的重要性并排、共享同一套特征标签（y 轴对齐），
#   一眼看出"哪些结论稳健、哪些是某种方法的偏好"。这是一图整合多来源信息。
# ============================================================================
def make_importance_triptych(ctx):
    section("高级组合图 B —— 特征重要性三合一")
    X = ctx["X"]
    X_test, y_test = ctx["X_test"], ctx["y_test"]
    rf = ctx["rf_bal"]
    labels = [FEATURE_CN.get(c, c) for c in X.columns]

    imp_gini = rf.feature_importances_
    perm = permutation_importance(rf, X_test, y_test, n_repeats=20,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    imp_perm = perm.importances_mean

    shap_mean = None
    if ctx.get("shap_values") is not None:
        shap_mean = np.abs(ctx["shap_values"]).mean(axis=0)

    # 统一按不纯度重要性排序，保证三个子图 y 轴顺序一致、便于横向比较
    order = np.argsort(imp_gini)
    yy = np.arange(len(order))
    ncol = 3 if shap_mean is not None else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 5.2), sharey=True)

    axes[0].barh(yy, imp_gini[order], color=C_TEAL)
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels([labels[i] for i in order])
    axes[0].set_title("(a) 不纯度重要性\n(训练集, 对连续变量有偏)")
    axes[0].set_xlabel("重要性")

    axes[1].barh(yy, imp_perm[order], xerr=perm.importances_std[order],
                 color=C_BLUE)
    axes[1].set_title("(b) 置换重要性\n(测试集, 衡量泛化贡献)")
    axes[1].set_xlabel("AUC/准确率下降")

    if shap_mean is not None:
        axes[2].barh(yy, shap_mean[order], color=C_PURPLE)
        axes[2].set_title("(c) 平均 |SHAP|\n(对预测的平均贡献幅度)")
        axes[2].set_xlabel("mean(|SHAP|)")

    plt.suptitle("高级组合图 B · 特征重要性三口径对照", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    savefig("10_高级_重要性三合一.png")
    plt.show(); plt.close()


# ============================================================================
#   高级组合大图 C —— 联合调参热力图 + 边际曲线（一图看清参数交互）
# ----------------------------------------------------------------------------
# 设计思路：主面板是 n_estimators × max_features 的 CV-AUC 热力图（看交互），
#   顶部与右侧各挂一条边际曲线（对另一维求均值），并在最优格子打星标。
#   这样把"逐一调参"的两张一维曲线与"联合搜索"的二维视图合成一张图。
# ============================================================================
def make_tuning_heatmap(ctx):
    section("高级组合图 C —— 联合调参热力图 + 边际曲线")
    X = ctx["X"]
    X_train, y_train = ctx["X_train"], ctx["y_train"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    n_list = list(range(50, 351, 50))
    mf_list = list(range(1, X.shape[1] + 1))
    grid = np.zeros((len(mf_list), len(n_list)))
    for i, mf in enumerate(mf_list):
        for j, n in enumerate(n_list):
            m = RandomForestClassifier(n_estimators=n, max_features=mf,
                                       random_state=RANDOM_STATE, n_jobs=-1)
            grid[i, j] = cross_val_score(m, X_train, y_train, cv=cv,
                                         scoring="roc_auc").mean()
    bi, bj = np.unravel_index(np.argmax(grid), grid.shape)

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=(5, 1.3), height_ratios=(1.3, 5),
                          wspace=0.05, hspace=0.05)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    im = ax_main.imshow(grid, aspect="auto", cmap=CMAP_SEQ, origin="lower")
    ax_main.set_xticks(range(len(n_list))); ax_main.set_xticklabels(n_list)
    ax_main.set_yticks(range(len(mf_list))); ax_main.set_yticklabels(mf_list)
    ax_main.set_xlabel("n_estimators"); ax_main.set_ylabel("max_features")
    ax_main.scatter([bj], [bi], marker="*", s=320, color=C_RED,
                    edgecolor="white", zorder=5,
                    label="最优(AUC=%.3f)" % grid[bi, bj])
    ax_main.legend(loc="lower right", fontsize=8)
    for i in range(len(mf_list)):
        for j in range(len(n_list)):
            ax_main.text(j, i, "%.3f" % grid[i, j], ha="center", va="center",
                         color="white", fontsize=7)

    # 顶部边际曲线：对 max_features 求均值，看 n_estimators 的边际影响
    ax_top.plot(range(len(n_list)), grid.mean(axis=0), "o-", color=C_RED)
    ax_top.axvline(bj, ls="--", color=C_RED)
    ax_top.set_ylabel("均值AUC"); ax_top.tick_params(labelbottom=False)
    ax_top.set_title("联合调参: 边际曲线挂在热力图两侧", fontsize=10)

    # 右侧边际曲线：对 n_estimators 求均值，看 max_features 的边际影响
    ax_right.plot(grid.mean(axis=1), range(len(mf_list)), "s-", color=C_BLUE)
    ax_right.axhline(bi, ls="--", color=C_RED)
    ax_right.set_xlabel("均值AUC"); ax_right.tick_params(labelleft=False)

    fig.colorbar(im, ax=ax_right, fraction=0.08, pad=0.25, label="CV AUC")
    plt.suptitle("高级组合图 C · n_estimators × max_features 联合搜索",
                 fontsize=14, fontweight="bold")
    savefig("11_高级_调参热力图.png")
    plt.show(); plt.close()


# ============================================================================
#   高级组合大图 D —— 决策边界 + 概率等高线 + 数据密度（三层信息叠一张图）
# ----------------------------------------------------------------------------
# 设计思路：仅取两个特征(workyears, debtratio)张成平面，同一张图叠三层信息：
#   底层=预测概率填充等高线(渐变，显示模型对空间的判断)，
#   中层=0.5 决策边界(粗线)，
#   顶层=真实样本散点(按真实类着色)。横纵坐标即两个特征，信息高度融合。
# 注意：这是为"可视化"服务的二维简化模型，非最终模型。
# ============================================================================
def make_boundary_plot(ctx):
    section("高级组合图 D —— 决策边界 + 概率等高线 + 数据密度")
    X, y = ctx["X"], ctx["y"]
    feats = ["workyears", "debtratio"]
    X2 = X[feats].values
    # 训练集拟合(避免原文用全量 fit 再自评的泄露)，此处仅为可视化
    Xtr, Xte, ytr, yte = train_test_split(
        X2, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
    mdl = RandomForestClassifier(n_estimators=300, max_features=1,
                                 class_weight="balanced",
                                 random_state=RANDOM_STATE, n_jobs=-1)
    mdl.fit(Xtr, ytr)

    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    prob = mdl.predict_proba(grid)[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 7))
    # 底层：概率填充等高线
    cf = ax.contourf(xx, yy, prob, levels=20, cmap=CMAP_RISK, alpha=0.75)
    # 中层：0.5 决策边界
    ax.contour(xx, yy, prob, levels=[0.5], colors=C_DARK, linewidths=2)
    # 顶层：真实样本散点
    ax.scatter(X2[y == 0, 0], X2[y == 0, 1], c=C_TEAL, s=20,
               edgecolor="white", linewidth=0.5, label="真实未违约")
    ax.scatter(X2[y == 1, 0], X2[y == 1, 1], c=C_MAUVE, s=22, marker="^",
               edgecolor=C_DARK, linewidth=0.5, label="真实违约")
    ax.set_xlabel("工作年限 workyears"); ax.set_ylabel("债务收入比 debtratio")
    ax.set_title("高级组合图 D · 决策边界+概率等高线+样本分布 三层合一",
                 fontweight="bold")
    ax.legend(loc="upper right")
    fig.colorbar(cf, ax=ax, label="预测为违约的概率")
    savefig("12_高级_决策边界组合图.png")
    plt.show(); plt.close()


# ============================================================================
#   高级组合大图 E —— 数据全景（类别占比 + 相关性 + 分箱违约率 + 8 特征分布）
# ----------------------------------------------------------------------------
# 设计思路：用非均匀 GridSpec 把 EDA 的 11 张小图合成一张。第一行看"全局"
#   （类别是否平衡、特征间是否共线、哪些特征与违约率单调相关），
#   后两行看"局部"（每个特征在两类上的分布差异），读者一张图即可建立数据直觉。
# ============================================================================
def make_eda_panorama(ctx):
    section("高级组合图 E —— 数据全景")
    X, y = ctx["X"], ctx["y"]
    cn = [FEATURE_CN.get(c, c) for c in X.columns]

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.42, wspace=0.32,
                          height_ratios=(1.25, 1, 1))

    # (a) 类别占比环形图
    ax = fig.add_subplot(gs[0, 0])
    vc = y.value_counts().sort_index()
    ax.pie(vc, labels=["未违约\n%d" % vc[0], "违约\n%d" % vc[1]],
           colors=[C_NEG, C_POS], autopct="%.1f%%", startangle=90,
           pctdistance=0.78, wedgeprops=dict(width=0.42, edgecolor="white"))
    ax.text(0, 0, "N=%d" % len(y), ha="center", va="center",
            fontsize=13, fontweight="bold", color=C_DARK)
    ax.set_title("(a) 类别占比")

    # (b) 特征相关性热力图（下三角 + 数值标注）
    ax = fig.add_subplot(gs[0, 1:3])
    corr = X.corr().values
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    im = ax.imshow(np.ma.masked_where(mask, corr), cmap=CMAP_DIV, vmin=-1, vmax=1)
    for i in range(len(cn)):
        for j in range(i + 1):
            ax.text(j, i, "%.2f" % corr[i, j], ha="center", va="center",
                    fontsize=8, color="white" if abs(corr[i, j]) > 0.6 else C_DARK)
    ax.set_xticks(range(len(cn))); ax.set_xticklabels(cn, rotation=35, ha="right")
    ax.set_yticks(range(len(cn))); ax.set_yticklabels(cn)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Pearson r")
    ax.set_title("(b) 特征相关性（下三角）")
    # (c) 十分位分箱违约率：看特征与违约的单调性
    ax = fig.add_subplot(gs[0, 3])
    for col, color in zip(["debtratio", "creditdebt", "workyears", "income"],
                          [C_RED, C_ORANGE, C_TEAL, C_BLUE]):
        bins = pd.qcut(X[col], 10, duplicates="drop")
        rate = y.groupby(bins, observed=True).mean()
        ax.plot(np.arange(1, len(rate) + 1), rate.values, "o-", ms=4,
                color=color, label=FEATURE_CN[col])
    ax.axhline(y.mean(), ls="--", color=C_DARK, lw=1, label="总体违约率")
    ax.set_xlabel("特征十分位(低→高)"); ax.set_ylabel("违约率")
    ax.set_title("(c) 分箱违约率"); ax.legend(fontsize=7)

    # (d)~(k) 8 个特征：两类重叠直方图 + 均值竖线
    for i, col in enumerate(X.columns):
        ax = fig.add_subplot(gs[1 + i // 4, i % 4])
        bins = np.histogram_bin_edges(X[col], bins=22)
        for cls, color, lab in [(0, C_NEG, "未违约"), (1, C_POS, "违约")]:
            v = X.loc[y == cls, col]
            ax.hist(v, bins=bins, density=True, alpha=0.55, color=color, label=lab)
            ax.axvline(v.mean(), color=color, ls="--", lw=1.5)
        ax.set_title("(%s) %s" % (chr(ord("d") + i), FEATURE_CN[col]), fontsize=10)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=8)

    plt.suptitle("高级组合图 E · 数据全景（类别 + 相关性 + 分箱违约率 + 分布差异）",
                 fontsize=15, fontweight="bold", y=0.995)
    finalize("13_高级_数据全景.png")


# ============================================================================
#   高级组合大图 F —— Top3 特征诊断矩阵（4 行视角 × 3 列特征）
# ----------------------------------------------------------------------------
# 设计思路：同一列 = 同一个特征，从上到下依次回答：
#   它长什么样(分布) → 两类是否可分(箱线) → 模型平均怎么用它(PDP+ICE)
#   → 对单个样本贡献多少(SHAP 依赖)。PDP/SHAP/分布首次在同一 x 轴上对齐。
# ============================================================================
def make_feature_diagnostic(ctx):
    section("高级组合图 F —— Top3 特征诊断矩阵")
    rf = ctx["rf_bal"]
    X_train, y_train = ctx["X_train"], ctx["y_train"]
    perm = permutation_importance(rf, ctx["X_test"], ctx["y_test"], n_repeats=10,
                                  scoring="roc_auc",
                                  random_state=RANDOM_STATE, n_jobs=-1)
    top3 = list(X_train.columns[np.argsort(perm.importances_mean)[::-1][:3]])
    print("  置换重要性 Top3:", [FEATURE_CN[c] for c in top3])
    has_shap = ctx.get("shap_values") is not None
    nrow = 4 if has_shap else 3

    fig, axes = plt.subplots(nrow, 3, figsize=(15, 3.3 * nrow))
    for j, col in enumerate(top3):
        x = X_train[col]
        # 行1：总体分布 + 核密度感的累计线
        ax = axes[0, j]
        ax.hist(x, bins=25, color=C_BLUEL, edgecolor=C_BLUE2)
        ax.set_title(FEATURE_CN[col], fontsize=12, fontweight="bold")
        ax2 = ax.twinx(); ax2.grid(False)
        xs = np.sort(x.values)
        ax2.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=C_DARK, lw=1)
        ax2.set_ylim(0, 1.05); ax2.tick_params(labelsize=7)

        # 行2：按类别箱线 + 抖动散点
        ax = axes[1, j]
        data_ = [x[y_train == 0], x[y_train == 1]]
        bp = ax.boxplot(data_, vert=False, widths=0.5, patch_artist=True,
                        showfliers=False)
        for patch, c in zip(bp["boxes"], [C_NEG, C_POS]):
            patch.set_facecolor(c); patch.set_alpha(0.5)
        rng = np.random.default_rng(RANDOM_STATE)
        for k, (d, c) in enumerate(zip(data_, [C_NEG, C_POS])):
            ax.scatter(d, k + 1 + rng.uniform(-0.18, 0.18, len(d)),
                       s=5, color=c, alpha=0.5)
        ax.set_yticks([1, 2]); ax.set_yticklabels(["未违约", "违约"])
        ax.sharex(axes[0, j])

        # 行3：PDP(粗线) + ICE(细线，抽 60 条)
        ax = axes[2, j]
        # 注意：partial_dependence 函数本身没有 subsample 参数，ICE 抽样需手动完成
        pdp = partial_dependence(rf, X_train, [col], kind="both",
                                 grid_resolution=40)
        grid_x = pdp["grid_values"][0] if "grid_values" in pdp else pdp["values"][0]
        ice = pdp["individual"][0]
        pick = np.random.default_rng(RANDOM_STATE).choice(
            len(ice), size=min(60, len(ice)), replace=False)
        for line in ice[pick]:
            ax.plot(grid_x, line, color=C_PURPLE, alpha=0.15, lw=0.7)
        ax.plot(grid_x, pdp["average"][0], color=C_RED, lw=2.5, label="PDP 平均")
        ax.set_ylim(0, 1); ax.legend(fontsize=8, loc="upper left")
        ax.sharex(axes[0, j])

        # 行4：SHAP 依赖图（颜色=交互最强的另一个特征）
        if has_shap:
            ax = axes[3, j]
            sv, Xs = ctx["shap_values"], ctx["shap_X"]
            k = list(Xs.columns).index(col)
            others = [c for c in Xs.columns if c != col]
            # 颜色特征 = 与该特征 SHAP 值相关性最强的另一特征（粗略的交互代理）
            inter = max(others, key=lambda c: np.nan_to_num(
                abs(np.corrcoef(Xs[c], sv[:, k])[0, 1])))
            sc = ax.scatter(Xs[col], sv[:, k], c=Xs[inter], cmap=CMAP_RISK,
                            s=18, edgecolor=C_DARK, linewidth=0.3)
            ax.axhline(0, color=C_DARK, lw=0.8)
            cb = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.05)
            cb.set_label(FEATURE_CN[inter], fontsize=8); cb.ax.tick_params(labelsize=7)
            ax.sharex(axes[0, j])
            ax.set_xlabel(FEATURE_CN[col])

    for i, label in enumerate(["分布 + 累计", "按类别分布", "PDP + ICE", "SHAP 依赖"][:nrow]):
        axes[i, 0].set_ylabel(label, fontsize=11, fontweight="bold")
    plt.suptitle("高级组合图 F · Top3 特征诊断矩阵（分布 → 可分性 → 平均效应 → 个体贡献）",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    finalize("14_高级_特征诊断矩阵.png")


# ============================================================================
#   高级组合大图 G —— 多模型横评六合一
# ----------------------------------------------------------------------------
# 设计思路：6 个模型在 6 种视角下同屏对比，全图共用"模型→颜色"映射，
#   视觉主线就是模型本身：ROC(排序) / PR(少数类) / 校准(概率可信度) /
#   雷达图(五指标综合) / 概率分离度(两类概率分布) / 累积捕获(业务覆盖)。
# ============================================================================
def make_model_showdown(ctx):
    section("高级组合图 G —— 多模型横评六合一")
    X_test, y_test = ctx["X_test"], ctx["y_test"]
    models, cmp_df = ctx["models"], ctx["cmp_df"]
    names = list(models)
    colors = dict(zip(names, [C_BLUE2, C_ORANGE, C_PURPLE, C_RED, C_TEAL, C_MAUVE]))
    probs = {n: m.predict_proba(X_test)[:, 1] for n, m in models.items()}
    yv = y_test.values

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

    # (1) ROC
    ax = fig.add_subplot(gs[0, 0])
    for n in names:
        fpr, tpr, _ = roc_curve(yv, probs[n])
        lw = 2.8 if n == "随机森林" else 1.4
        ax.plot(fpr, tpr, color=colors[n], lw=lw,
                label="%s  %.3f" % (n, roc_auc_score(yv, probs[n])))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("(1) ROC 曲线 (图例=AUC)"); ax.legend(fontsize=8, loc="lower right")

    # (2) PR
    ax = fig.add_subplot(gs[0, 1])
    for n in names:
        prec, rec, _ = precision_recall_curve(yv, probs[n])
        ax.plot(rec, prec, color=colors[n], lw=2.8 if n == "随机森林" else 1.4,
                label="%s  %.3f" % (n, average_precision_score(yv, probs[n])))
    ax.axhline(yv.mean(), ls="--", color=C_DARK, lw=1)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("(2) PR 曲线 (图例=AP)"); ax.legend(fontsize=8, loc="upper right")

    # (3) 校准 + Brier
    ax = fig.add_subplot(gs[0, 2])
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for n in names:
        fp, mp = calibration_curve(yv, probs[n], n_bins=8, strategy="quantile")
        brier = np.mean((probs[n] - yv) ** 2)
        ax.plot(mp, fp, "o-", ms=4, color=colors[n], label="%s  %.3f" % (n, brier))
    ax.set_xlabel("预测概率"); ax.set_ylabel("实际违约比例")
    ax.set_title("(3) 校准曲线 (图例=Brier, 越小越好)"); ax.legend(fontsize=8)

    # (4) 五指标雷达图
    ax = fig.add_subplot(gs[1, 0], projection="polar")
    metrics = ["准确率", "AUC", "AP", "F1", "Kappa"]
    ang = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    ang += ang[:1]
    for n in names:
        v = cmp_df.loc[n, metrics].tolist(); v += v[:1]
        ax.plot(ang, v, color=colors[n], lw=2.5 if n == "随机森林" else 1.2, label=n)
        if n == "随机森林":
            ax.fill(ang, v, color=colors[n], alpha=0.15)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1); ax.set_title("(4) 五指标雷达(默认阈值0.5)", pad=16)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.35, 1.12))

    # (5) 概率分离度：每个模型两类概率的小提琴
    ax = fig.add_subplot(gs[1, 1])
    pos = np.arange(len(names))
    for cls, off, c in [(0, -0.18, C_NEG), (1, 0.18, C_POS)]:
        vp = ax.violinplot([probs[n][yv == cls] for n in names], positions=pos + off,
                           widths=0.34, showmedians=True)
        for b in vp["bodies"]:
            b.set_facecolor(c); b.set_alpha(0.55); b.set_edgecolor(C_DARK)
        for k in ("cbars", "cmins", "cmaxes", "cmedians"):
            vp[k].set_color(C_DARK); vp[k].set_linewidth(0.8)
    for i, n in enumerate(names):
        ks = np.max(np.abs(roc_curve(yv, probs[n])[1] - roc_curve(yv, probs[n])[0]))
        ax.text(i, 1.04, "KS=%.2f" % ks, ha="center", fontsize=8, color=C_DARK)
    ax.set_xticks(pos); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(-0.02, 1.12); ax.set_ylabel("预测违约概率")
    ax.scatter([], [], color=C_NEG, label="真实未违约")
    ax.scatter([], [], color=C_POS, label="真实违约")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.97))
    ax.set_title("(5) 两类概率分离度")

    # (6) 累积捕获曲线 (Gain)
    ax = fig.add_subplot(gs[1, 2])
    xs = np.arange(1, len(yv) + 1) / len(yv)
    for n in names:
        cum = np.cumsum(yv[np.argsort(-probs[n])]) / yv.sum()
        ax.plot(xs, cum, color=colors[n], lw=2.8 if n == "随机森林" else 1.4,
                label="%s  前20%%捕获%.0f%%" % (n, cum[int(0.2 * len(yv)) - 1] * 100))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.axvline(0.2, color=C_GRAY, lw=1)
    ax.set_xlabel("按风险从高到低覆盖的样本比例"); ax.set_ylabel("捕获的违约比例")
    ax.set_title("(6) 累积捕获曲线"); ax.legend(fontsize=8, loc="lower right")

    plt.suptitle("高级组合图 G · 多模型横评六合一（排序 / 少数类 / 校准 / 综合 / 分离 / 覆盖）",
                 fontsize=15, fontweight="bold", y=0.995)
    finalize("15_高级_多模型横评六合一.png")


def _beeswarm(ax, sv, Xs, labels, order):
    """手绘 SHAP 蜂群图：可放进任意子图（shap.summary_plot 只能独占 figure）。"""
    rng = np.random.default_rng(RANDOM_STATE)
    for row, k in enumerate(order):
        v = Xs.iloc[:, k].values
        vn = (v - v.min()) / (np.ptp(v) + 1e-12)          # 特征值归一化着色
        jitter = rng.normal(0, 0.09, len(v))
        sc = ax.scatter(sv[:, k], row + jitter, c=vn, cmap=CMAP_RISK,
                        s=10, edgecolor="none", alpha=0.85)
    ax.axvline(0, color=C_DARK, lw=0.8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([labels[k] for k in order])
    return sc


def _waterfall(ax, sv_row, x_row, base, labels, title, max_show=8):
    """手绘 SHAP 瀑布图：从基准值出发逐特征累加到最终预测。"""
    order = np.argsort(np.abs(sv_row))[::-1][:max_show][::-1]
    cur = base + sv_row.sum() - sv_row[order].sum()       # 未显示特征合并进起点
    for row, k in enumerate(order):
        v = sv_row[k]
        ax.barh(row, v, left=cur, color=C_POS if v > 0 else C_NEG,
                edgecolor="white", height=0.7)
        ax.text(cur + v + (0.004 if v > 0 else -0.004), row, "%+.3f" % v,
                va="center", ha="left" if v > 0 else "right", fontsize=8)
        cur += v
    ax.axvline(base, color=C_DARK, ls="--", lw=1)
    ax.axvline(cur, color=C_RED, lw=1.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(["%s = %g" % (labels[k], round(x_row[k], 2)) for k in order],
                       fontsize=9)
    ax.set_xlabel("违约概率  (虚线=基准 %.3f, 红线=预测 %.3f)" % (base, cur))
    ax.set_title(title)


# ============================================================================
#   高级组合大图 H —— SHAP 深度解析（全局 + 交互 + 两个典型个体）
# ----------------------------------------------------------------------------
# 设计思路：左上 蜂群(重要性+方向)、右上 |SHAP| 条形 + 正/负贡献拆分，
#   下排两张瀑布图：最"确定违约"与最"确定不违约"的样本各一。
#   全局与个体解释放同一画布对读，回答"整体看什么、具体为什么"。
# ============================================================================
def make_shap_deep_dive(ctx):
    section("高级组合图 H —— SHAP 深度解析")
    if ctx.get("shap_values") is None:
        print("  [跳过] 无 SHAP 值。"); return
    sv, Xs = ctx["shap_values"], ctx["shap_X"]
    base = ctx.get("shap_base", float(ctx["y_train"].mean()))
    labels = [FEATURE_CN.get(c, c) for c in Xs.columns]
    mean_abs = np.abs(sv).mean(0)
    order = np.argsort(mean_abs)                         # 升序 → 最重要在上

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.42, width_ratios=(1.25, 1))

    ax = fig.add_subplot(gs[0, 0])
    sc = _beeswarm(ax, sv, Xs, labels, order)
    cb = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.04)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["低", "高"]); cb.set_label("特征值")
    ax.set_xlabel("SHAP 值 (对违约概率的影响)")
    ax.set_title("(a) 蜂群图：每个点是一个样本")

    ax = fig.add_subplot(gs[0, 1])
    pos_c = np.where(sv > 0, sv, 0).mean(0)
    neg_c = np.where(sv < 0, sv, 0).mean(0)
    yy = np.arange(len(order))
    ax.barh(yy, pos_c[order], color=C_POS, label="推高违约(+)")
    ax.barh(yy, neg_c[order], color=C_NEG, label="拉低违约(-)")
    ax.scatter(mean_abs[order], yy, marker="D", color=C_DARK, s=28, zorder=5,
               label="mean|SHAP|")
    ax.axvline(0, color=C_DARK, lw=0.8)
    ax.set_yticks(yy); ax.set_yticklabels([labels[k] for k in order])
    ax.set_xlabel("平均贡献"); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("(b) 重要性拆分：正向 vs 负向贡献")

    pred = base + sv.sum(1)
    i_hi, i_lo = int(np.argmax(pred)), int(np.argmin(pred))
    y_true = ctx["y_test"].loc[Xs.index].values
    for slot, idx, tag in [(gs[1, 0], i_hi, "最高风险样本"), (gs[1, 1], i_lo, "最低风险样本")]:
        ax = fig.add_subplot(slot)
        _waterfall(ax, sv[idx], Xs.iloc[idx].values, base, labels,
                   "(%s) %s  #%d  真实=%s" % ("c" if idx == i_hi else "d", tag,
                                             Xs.index[idx],
                                             "违约" if y_true[idx] == 1 else "未违约"))

    plt.suptitle("高级组合图 H · SHAP 深度解析（全局重要性 + 方向 + 个体归因）",
                 fontsize=15, fontweight="bold", y=0.995)
    finalize("16_高级_SHAP深度解析.png")


# ============================================================================
#   高级组合大图 I —— 业务决策画布（代价 / 阈值 / KS / 提升 / 分层 / 审批）
# ----------------------------------------------------------------------------
# 设计思路：把"模型分数怎么变成业务决策"一张图讲清楚。
#   阈值在训练集 5 折 OOF 概率上选（不碰测试集），在测试集上展示效果。
#   代价假设：漏判违约(FN)=5，误拒好客户(FP)=1，可按业务修改 COST_FN/COST_FP。
# ============================================================================
COST_FN, COST_FP = 5.0, 1.0


def make_risk_decision_canvas(ctx):
    section("高级组合图 I —— 业务决策画布")
    rf = ctx["rf_bal"]
    X_train, y_train = ctx["X_train"], ctx["y_train"]
    yv, prob = ctx["y_test"].values, ctx["prob_test"]

    # OOF 概率：每个训练样本由"没见过它"的模型打分，用来无泄露地选阈值
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(rf, X_train, y_train, cv=cv, method="predict_proba",
                            n_jobs=-1)[:, 1]
    ths = np.linspace(0.02, 0.98, 97)

    def cost_curve(y_, p_):
        return np.array([(COST_FN * ((p_ < t) & (y_ == 1)).sum() +
                          COST_FP * ((p_ >= t) & (y_ == 0)).sum()) / len(y_)
                         for t in ths])
    cost_oof, cost_te = cost_curve(y_train.values, oof), cost_curve(yv, prob)
    t_cost = ths[int(np.argmin(cost_oof))]
    t_f1 = ths[int(np.argmax([f1_score(y_train, (oof >= t).astype(int)) for t in ths]))]
    ctx["t_cost"] = t_cost
    print("  OOF 代价最优阈值=%.2f, OOF F1最优阈值=%.2f" % (t_cost, t_f1))

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.33, wspace=0.28)

    # (1) 期望代价-阈值
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ths, cost_oof, color=C_BLUE, lw=2, label="训练OOF(用于选阈值)")
    ax.plot(ths, cost_te, color=C_RED, lw=1.5, ls="-.", label="测试集(仅展示)")
    ax.axvline(t_cost, color=C_DARK, ls="--", label="代价最优 %.2f" % t_cost)
    ax.axvline(t_f1, color=C_ORANGE, ls=":", lw=2, label="F1最优 %.2f" % t_f1)
    ax.axvline(0.5, color=C_GRAY, lw=1.5, label="默认 0.50")
    ax.set_xlabel("分类阈值"); ax.set_ylabel("人均期望代价")
    ax.set_title("(1) 期望代价曲线 (FN=%g, FP=%g)" % (COST_FN, COST_FP))
    ax.legend(fontsize=8)

    # (2) 三种阈值下的混淆矩阵并排（嵌套 GridSpec）
    sub = gs[0, 1].subgridspec(1, 3, wspace=0.08)
    for k, (t, nm) in enumerate([(0.5, "默认"), (t_f1, "F1最优"), (t_cost, "代价最优")]):
        a = fig.add_subplot(sub[0, k])
        cm = confusion_matrix(yv, (prob >= t).astype(int))
        a.imshow(cm, cmap=CMAP_BLUE); a.grid(False)
        for i in range(2):
            for j in range(2):
                a.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
        c = (COST_FN * cm[1, 0] + COST_FP * cm[0, 1]) / cm.sum()
        a.set_title("%s t=%.2f\n代价=%.3f" % (nm, t, c), fontsize=9)
        a.set_xticks([0, 1]); a.set_xticklabels(["预测0", "预测1"], fontsize=8)
        a.set_yticks([0, 1]); a.set_yticklabels(["真0", "真1"], fontsize=8)
        if k > 0:
            a.tick_params(labelleft=False)
    fig.text(0.5, 0.945, "(2) 三种阈值下的测试集混淆矩阵", ha="center", fontsize=12)

    # (3) KS 曲线
    ax = fig.add_subplot(gs[0, 2])
    ps = np.sort(np.unique(prob))
    cdf_bad = [(prob[yv == 1] <= s).mean() for s in ps]
    cdf_good = [(prob[yv == 0] <= s).mean() for s in ps]
    diff = np.array(cdf_good) - np.array(cdf_bad)
    ki = int(np.argmax(diff))
    ax.plot(ps, cdf_good, color=C_NEG, lw=2, label="未违约 CDF")
    ax.plot(ps, cdf_bad, color=C_POS, lw=2, label="违约 CDF")
    ax.vlines(ps[ki], cdf_bad[ki], cdf_good[ki], color=C_DARK, lw=2.5)
    ax.text(ps[ki] + 0.02, (cdf_bad[ki] + cdf_good[ki]) / 2,
            "KS=%.3f\n@%.2f" % (diff[ki], ps[ki]), fontsize=10, fontweight="bold")
    ax.set_xlabel("预测违约概率"); ax.set_ylabel("累计比例")
    ax.set_title("(3) KS 曲线 (两类累计分布最大差)"); ax.legend(fontsize=8)

    # (4) 十分位风险分层：样本数(柱) + 实际违约率(线) + 提升度
    ax = fig.add_subplot(gs[1, 0:2])
    dec = pd.qcut(pd.Series(prob).rank(method="first"), 10, labels=False)
    dec = 9 - dec                                          # 0=最高风险组
    tab = pd.DataFrame({"d": dec, "y": yv, "p": prob}).groupby("d").agg(
        n=("y", "size"), bad=("y", "sum"), rate=("y", "mean"), pm=("p", "mean"))
    tab["lift"] = tab["rate"] / yv.mean()
    tab["cum_capture"] = tab["bad"].cumsum() / yv.sum()
    xl = np.arange(10)
    ax.bar(xl - 0.2, tab["n"] - tab["bad"], 0.4, color=C_NEG, alpha=0.7, label="未违约人数")
    ax.bar(xl + 0.2, tab["bad"], 0.4, color=C_POS, alpha=0.85, label="违约人数")
    ax.set_ylabel("人数"); ax.set_xticks(xl)
    ax.set_xticklabels(["D%d\n%.2f~" % (i + 1, prob[dec.values == i].min()) for i in xl],
                       fontsize=8)
    ax.set_xlabel("风险十分位 (D1=最高风险, 下方为该组最低分)")
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(xl, tab["rate"], "o-", color=C_RED, lw=2, label="实际违约率")
    ax2.plot(xl, tab["pm"], "s--", color=C_PURPLE, lw=1.5, label="平均预测概率")
    ax2.plot(xl, tab["cum_capture"], "^-", color=C_BLUE, lw=1.5, label="累计捕获率")
    for i in xl:
        ax2.text(i, tab["rate"].iloc[i] + 0.04, "Lift %.1f" % tab["lift"].iloc[i],
                 ha="center", fontsize=8, color=C_RED)
    ax2.set_ylim(0, 1.15); ax2.set_ylabel("比例")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center", ncol=5)
    ax.set_title("(4) 风险十分位分层表 (柱=人数, 线=违约率/预测/累计捕获)")

    # (5) 审批策略：通过率 vs 通过客群坏账率
    ax = fig.add_subplot(gs[1, 2])
    approve = np.array([(prob < t).mean() for t in ths])
    badrate = np.array([yv[prob < t].mean() if (prob < t).any() else np.nan for t in ths])
    sc = ax.scatter(approve, badrate, c=ths, cmap=CMAP_RISK, s=22, edgecolor=C_DARK,
                    linewidth=0.3)
    for t, mk, nm in [(0.5, "o", "默认"), (t_f1, "s", "F1"), (t_cost, "*", "代价")]:
        i = int(np.argmin(np.abs(ths - t)))
        ax.scatter(approve[i], badrate[i], marker=mk, s=180, facecolor="none",
                   edgecolor=C_DARK, lw=2)
        ax.annotate("%s t=%.2f\n通过%.0f%% 坏账%.1f%%" % (nm, t, approve[i] * 100,
                                                     badrate[i] * 100),
                    (approve[i], badrate[i]), textcoords="offset points",
                    xytext=(-95, 10), fontsize=8)
    ax.axhline(yv.mean(), color=C_GRAY, ls="--", lw=1)
    fig.colorbar(sc, ax=ax, pad=0.01, label="阈值")
    ax.set_xlabel("审批通过率 (预测概率<阈值即通过)"); ax.set_ylabel("通过客群违约率")
    ax.set_title("(5) 审批策略曲线")

    plt.suptitle("高级组合图 I · 业务决策画布（代价 → 阈值 → 区分度 → 分层 → 审批）",
                 fontsize=15, fontweight="bold", y=0.995)
    finalize("17_高级_业务决策画布.png")


# ============================================================================
#   高级组合大图 J —— 误分类诊断（模型在哪儿错、错的人像谁）
# ----------------------------------------------------------------------------
# 设计思路：按测试集预测结果把样本切成 TP/FN/FP/TN 四组，从五个角度对比：
#   组规模、二维特征空间分布、标准化特征画像(雷达+热力)、预测概率位置。
#   FN(漏判违约)常常"长得像好客户"，这张图能直观看出来。
# ============================================================================
def make_error_diagnosis(ctx):
    section("高级组合图 J —— 误分类诊断")
    X_test, yv, prob = ctx["X_test"], ctx["y_test"].values, ctx["prob_test"]
    t = ctx.get("t_cost", ctx.get("best_threshold", 0.5))
    pred = (prob >= t).astype(int)
    grp = np.select([(yv == 1) & (pred == 1), (yv == 1) & (pred == 0),
                     (yv == 0) & (pred == 1)], ["TP", "FN", "FP"], "TN")
    G = ["TP", "FN", "FP", "TN"]
    GCN = {"TP": "TP 正确识别违约", "FN": "FN 漏判违约", "FP": "FP 误拒好客户",
           "TN": "TN 正确放行"}
    GC = {"TP": C_RED, "FN": C_ORANGE, "FP": C_PURPLE, "TN": C_TEAL}
    GM = {"TP": "^", "FN": "X", "FP": "D", "TN": "o"}
    labels = [FEATURE_CN.get(c, c) for c in X_test.columns]
    Z = (X_test - X_test.mean()) / X_test.std()          # 标准化后便于跨特征比较
    zmean = pd.DataFrame({g: Z[grp == g].mean() for g in G}).T

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.3)

    # (a) 四组规模 + 组内平均预测概率
    ax = fig.add_subplot(gs[0, 0])
    cnt = [int((grp == g).sum()) for g in G]
    bars = ax.bar(G, cnt, color=[GC[g] for g in G], edgecolor=C_DARK)
    for b, g, c in zip(bars, G, cnt):
        pm = prob[grp == g].mean() if c else np.nan
        ax.text(b.get_x() + b.get_width() / 2, c + 1, "%d\n均概率%.2f" % (c, pm),
                ha="center", fontsize=9)
    ax.set_ylim(0, max(cnt) * 1.25); ax.set_ylabel("样本数")
    rec = cnt[0] / max(cnt[0] + cnt[1], 1); pre = cnt[0] / max(cnt[0] + cnt[2], 1)
    ax.set_title("(a) 四组规模 (阈值=%.2f, Recall=%.2f, Precision=%.2f)" % (t, rec, pre),
                 fontsize=10)

    # (b) 二维特征空间分布（取 SHAP/置换最常见的两个关键特征）
    ax = fig.add_subplot(gs[0, 1])
    fx, fy = "debtratio", "workyears"
    for g in ["TN", "TP", "FP", "FN"]:                    # 错误样本后画，压在上层
        m = grp == g
        ax.scatter(X_test.loc[m, fx], X_test.loc[m, fy], marker=GM[g], color=GC[g],
                   s=30 + 70 * prob[m], edgecolor=C_DARK, linewidth=0.4,
                   alpha=0.9 if g in ("FN", "FP") else 0.45, label=GCN[g])
    ax.set_xlabel(FEATURE_CN[fx]); ax.set_ylabel(FEATURE_CN[fy])
    ax.set_title("(b) 关键特征平面 (点大小=预测概率)"); ax.legend(fontsize=8)

    # (c) 雷达画像
    ax = fig.add_subplot(gs[0, 2], projection="polar")
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist(); ang += ang[:1]
    for g in G:
        v = zmean.loc[g].tolist(); v += v[:1]
        ax.plot(ang, v, color=GC[g], lw=2, label=g)
        ax.fill(ang, v, color=GC[g], alpha=0.08)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("(c) 四组特征画像 (标准化均值)", pad=18)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1.12))

    # (d) 标准化均值热力图（横跨两列）+ FN vs TP 差值行
    ax = fig.add_subplot(gs[1, 0:2])
    mat = zmean.copy()
    mat.loc["FN-TP"] = zmean.loc["FN"] - zmean.loc["TP"]
    mat.loc["FP-TN"] = zmean.loc["FP"] - zmean.loc["TN"]
    lim = np.nanmax(np.abs(mat.values))
    im = ax.imshow(mat.values, cmap=CMAP_DIV, vmin=-lim, vmax=lim, aspect="auto")
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "%+.2f" % mat.values[i, j], ha="center", va="center",
                    fontsize=9, color=C_DARK)
    ax.axhline(3.5, color=C_DARK, lw=2)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([GCN.get(r, r + " (差异)") for r in mat.index])
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03, label="z 分数")
    ax.set_title("(d) 特征画像热力图：下两行 = 错误组相对正确组的偏移")

    # (e) 预测概率在四组中的位置
    ax = fig.add_subplot(gs[1, 2])
    bins = np.linspace(0, 1, 26)
    bottom = np.zeros(len(bins) - 1)
    for g in G:
        h, _ = np.histogram(prob[grp == g], bins=bins)
        ax.bar(bins[:-1], h, width=np.diff(bins), bottom=bottom, align="edge",
               color=GC[g], edgecolor="white", label=g)
        bottom += h
    ax.axvline(t, color=C_DARK, ls="--", lw=2, label="阈值 %.2f" % t)
    ax.set_xlabel("预测违约概率"); ax.set_ylabel("样本数")
    ax.set_title("(e) 堆叠概率直方图 (错误集中在阈值附近?)"); ax.legend(fontsize=8)

    plt.suptitle("高级组合图 J · 误分类诊断（规模 / 空间 / 画像 / 概率位置）",
                 fontsize=15, fontweight="bold", y=0.995)
    finalize("18_高级_误分类诊断.png")


# ============================================================================
#                              主流程
# ============================================================================
def main():
    print("随机森林分类算法 · 三级递进完整脚本")
    print("图片输出目录:", FIG_DIR)
    ctx = run_basic()
    ctx = run_intermediate(ctx)
    ctx = run_advanced(ctx)
    make_dashboard(ctx)
    make_importance_triptych(ctx)
    make_tuning_heatmap(ctx)
    make_boundary_plot(ctx)
    # ---- 新增：高信息密度融合大图 E~J ----
    make_eda_panorama(ctx)           # 13 数据全景
    make_feature_diagnostic(ctx)     # 14 Top3 特征诊断矩阵
    make_model_showdown(ctx)         # 15 多模型横评六合一
    make_shap_deep_dive(ctx)         # 16 SHAP 深度解析
    make_risk_decision_canvas(ctx)   # 17 业务决策画布（会写入 ctx["t_cost"]）
    make_error_diagnosis(ctx)        # 18 误分类诊断（使用代价最优阈值）

    # ---- 汇总输出 ----
    section("全部完成 —— 输出汇总")
    if "cmp_df" in ctx:
        print("模型横向对比表:\n", ctx["cmp_df"])
    print("\n所有图片已保存至:", FIG_DIR)
    for f in sorted(os.listdir(FIG_DIR)):
        print("   -", f)


if __name__ == "__main__":
    main()
