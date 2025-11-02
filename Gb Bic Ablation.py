# -*- coding: utf-8 -*-
"""
GB_BIC_ablation.py
在原始 GB-BIC 粒球聚类算法上整合“分裂准则消融实验”（BIC / AIC / NO_TEST / BIC_DELTA）。
- 入口会默认对 2d-10c.xlsx 批量运行四种配置并保存 ablation_split_results.csv
"""

from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.metrics import (
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    rand_score,
)
from scipy.optimize import linear_sum_assignment

# =========================

# =========================
CFG = {
    "split_rule": "BIC",     # ["BIC", "AIC", "NO_TEST", "BIC_DELTA"]
    "bic_delta": 9.0,         # 仅当 split_rule == "BIC_DELTA" 时生效
    "save_results": True,
}

# ============== 基础函数 ==============
def pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A.astype(float)
    B = B.astype(float)
    A2 = np.sum(A * A, axis=1, keepdims=True)
    B2 = np.sum(B * B, axis=1, keepdims=True).T
    return np.maximum(A2 + B2 - 2 * A.dot(B.T), 0.0)


def kmeans_two_clusters(X: np.ndarray, max_iter: int = 30):
    X = X.astype(float)
    n, d = X.shape
    rng = np.random.default_rng(42)
    i0 = rng.integers(0, n)
    c0 = X[i0]
    d2 = np.sum((X - c0) ** 2, axis=1)
    i1 = int(np.argmax(d2))
    centers = np.vstack([c0, X[i1]]).astype(float)

    labels = np.zeros(n, dtype=int)
    for it in range(max_iter):
        dmat_sq = pairwise_sq_dists(X, centers)
        new_labels = np.argmin(dmat_sq, axis=1)
        if np.array_equal(new_labels, labels) and it > 0:
            break
        labels = new_labels
        for k in range(2):
            pts = X[labels == k]
            if len(pts) > 0:
                centers[k] = pts.mean(axis=0)
    return labels, centers


def compute_stability_index(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    X = X.astype(float)
    centers = centers.astype(float)
    distances = np.linalg.norm(X - centers[labels], axis=1)
    intra_means = [distances[labels == k].mean() for k in np.unique(labels)]
    intra = np.mean(intra_means) if intra_means else 0.0
    inter = np.linalg.norm(centers[0] - centers[1]) if len(centers) > 1 else 0.0
    return inter / (intra + 1e-12)


# ============== 1D GMM + (AIC/BIC) ==============
def log_normal_pdf(y, mu, var):
    var = np.maximum(var, 1e-12)
    return -0.5 * np.log(2 * np.pi * var) - 0.5 * (y - mu) ** 2 / var


def gmm1d_em(y, max_iter=50, tol=1e-5, init=(None, None, None)):
    n = len(y)
    y = y.reshape(-1, 1).astype(float)
    if init[0] is None:
        m1, m2 = np.percentile(y, [30, 70])
        v = np.var(y) + 1e-6
        v1 = v2 = v
        pi = 0.5
    else:
        pi, (m1, v1), (m2, v2) = init

    ll_old = -np.inf
    for _ in range(max_iter):
        logp1 = np.log(pi + 1e-12) + log_normal_pdf(y, m1, v1)
        logp2 = np.log(1 - pi + 1e-12) + log_normal_pdf(y, m2, v2)
        mmax = np.maximum(logp1, logp2)
        p1 = np.exp(logp1 - mmax)
        p2 = np.exp(logp2 - mmax)
        gamma1 = p1 / (p1 + p2 + 1e-12)
        gamma2 = 1.0 - gamma1

        N1 = np.sum(gamma1)
        N2 = np.sum(gamma2)
        pi = N1 / (N1 + N2 + 1e-12)
        m1 = float(np.sum(gamma1 * y) / (N1 + 1e-12))
        m2 = float(np.sum(gamma2 * y) / (N2 + 1e-12))
        v1 = float(np.sum(gamma1 * (y - m1) ** 2) / (N1 + 1e-12)) + 1e-9
        v2 = float(np.sum(gamma2 * (y - m2) ** 2) / (N2 + 1e-12)) + 1e-9

        ll = np.sum(np.log(np.exp(logp1 - mmax) + np.exp(logp2 - mmax) + 1e-12) + mmax)
        if np.abs(ll - ll_old) < tol * (1 + np.abs(ll_old)):
            break
        ll_old = ll
    return ll, pi, m1, v1, m2, v2


def aic_bic_gmm1_vs_2(y, criterion="BIC", init_means=None, delta=0):
    """统一的 AIC/BIC 判定函数。
    - 返回 (stat1, stat2, accept)，其中 stat 为所选准则的数值；accept 表示是否接受 2 模型。
    - 参数个数近似：1D-GMM, K分量 => 自由度 ~ (K-1)[mixing] + K[means] + K[variances]
    """
    y = y.ravel().astype(float)
    n = len(y)
    if n < 4:
        return np.inf, np.inf, False

    # 单峰（K=1）
    mu1 = np.mean(y)
    var1 = np.var(y) + 1e-9
    ll1 = np.sum(log_normal_pdf(y, mu1, var1))
    k1 = (1 - 1) + 1 + 1  # 0 + mean + var = 2
    aic1 = 2 * k1 - 2 * ll1
    bic1 = k1 * np.log(n) - 2 * ll1

    # 双峰（K=2）
    if init_means is None:
        init = (None, None, None)
    else:
        m1i, m2i = init_means
        v = np.var(y) + 1e-6
        init = (0.5, (m1i, v), (m2i, v))
    ll2, _, _, _, _, _ = gmm1d_em(y, init=init)
    k2 = (2 - 1) + 2 + 2  # 1 + 2 + 2 = 5
    aic2 = 2 * k2 - 2 * ll2
    bic2 = k2 * np.log(n) - 2 * ll2

    if criterion == "AIC":
        return aic1, aic2, (aic2 < aic1)
    elif criterion == "BIC_DELTA":
        return bic1, bic2, (bic2 + float(delta) < bic1)
    elif criterion == "NO_TEST":
        return bic1, bic2, True
    else:  # 默认 BIC
        return bic1, bic2, (bic2 < bic1)


def check_split_accept(y_proj, mean1, mean2):
    rule = CFG["split_rule"]
    delta = CFG["bic_delta"] if rule == "BIC_DELTA" else 0.0
    _, _, accept = aic_bic_gmm1_vs_2(y_proj, criterion=rule, init_means=(mean1, mean2), delta=delta)
    return accept


# ============== 粒球生成（分裂） ==============
def generate_granular_balls(X: np.ndarray, verbose: bool = True):
    X = X.astype(float)
    n = len(X)
    queue = deque([np.arange(n)])
    clusters = []
    stability_history = []

    while queue:
        idx = queue.popleft()
        m = len(idx)
        cluster_points = X[idx]
        if m < 4:
            clusters.append(idx)
            continue

        labels, centers2 = kmeans_two_clusters(cluster_points)
        if len(np.unique(labels)) < 2:
            clusters.append(idx)
            continue

        stability_parent = compute_stability_index(cluster_points, labels, centers2)
        stability_history.append(stability_parent)
        S_med = np.median(stability_history)
        min_size_local = max(2, int(np.ceil(np.sqrt(m) / (stability_parent / (S_med + 1e-12) + 1e-12))))

        sub1 = idx[labels == 0]
        sub2 = idx[labels == 1]
        if len(sub1) < min_size_local or len(sub2) < min_size_local:
            clusters.append(idx)
            continue

        # 1D 投影（连心线）
        v = (centers2[1] - centers2[0]).astype(float)
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            clusters.append(idx)
            continue
        v = v / nv
        y = cluster_points @ v
        mean1 = y[labels == 0].mean()
        mean2 = y[labels == 1].mean()

        # === 分裂判定（已替换为通用 AIC/BIC 函数） ===
        accept = check_split_accept(y, mean1, mean2)

        if accept:
            queue.append(sub1)
            queue.append(sub2)
        else:
            clusters.append(idx)

    if verbose:
        tag = CFG["split_rule"] if CFG.get("split_rule") else "BIC"
        if tag == "BIC_DELTA":
            tag += f"(Δ={CFG.get('bic_delta', 0)})"
        print(f"Generated {len(clusters)} granular balls ({tag} splits).")
    return clusters


# ============== 小簇合并规则 ==============
def merge_small_clusters(X: np.ndarray, clusters):
    X = X.astype(float)
    N = len(X)
    min_size = int(np.sqrt(N))
    big_clusters = [set(idx) for idx in clusters if len(idx) >= min_size]
    small_clusters = [set(idx) for idx in clusters if len(idx) < min_size]

    if len(big_clusters) == 0:
        return small_clusters

    big_centers = np.array([X[list(c)].mean(axis=0) for c in big_clusters])
    small_centers = np.array([X[list(s)].mean(axis=0) for s in small_clusters]) if small_clusters else np.empty((0, X.shape[1]))

    if len(small_centers) > 0:
        diff = small_centers[:, None, :] - big_centers[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        nearest_big = dist_matrix.argmin(axis=1)
        for i, s in enumerate(small_clusters):
            j = nearest_big[i]
            big_clusters[j] |= s
    return big_clusters


# ============== 辅助函数：生成标签 & ACC计算 ==============
def cluster_labels_from_sets(clusters, n):
    labels = np.empty(n, dtype=int)
    for k, cluster in enumerate(clusters):
        for i in cluster:
            labels[i] = k
    return labels


def acc_score(y_true, y_pred):
    y_true = y_true.astype(int)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size


# ============== 主运行流程（返回指标字典） ==============
def run(data_file: str):
    data_file = data_file.strip()
    t0 = time.time()
    if data_file.endswith(".xlsx"):
        df = pd.read_excel(data_file)
    else:
        df = pd.read_csv(data_file)
    X = df.values
    print("Data shape:", X.shape)

    clusters = generate_granular_balls(X, verbose=True)
    merged_clusters = merge_small_clusters(X, clusters)

    n, d = X.shape
    labels_pred = cluster_labels_from_sets(merged_clusters, n)

    results = {}
    if d >= 3:  # 有标签（默认最后一列为标签）
        y_true = X[:, -1].astype(int)
        nmi = normalized_mutual_info_score(y_true, labels_pred)
        acc = acc_score(y_true, labels_pred)
        ari = adjusted_rand_score(y_true, labels_pred)
        ri = rand_score(y_true, labels_pred)
        results.update({"ACC": float(acc), "NMI": float(nmi), "ARI": float(ari), "RI": float(ri)})
        print(f"[有标签评价] ACC={acc:.4f}, NMI={nmi:.4f}, ARI={ari:.4f}, RI={ri:.4f}")
    else:       # 无标签
        X_feat = X
        sil = silhouette_score(X_feat, labels_pred)
        dbi = davies_bouldin_score(X_feat, labels_pred)
        ch = calinski_harabasz_score(X_feat, labels_pred)
        results.update({"Silhouette": float(sil), "DBI": float(dbi), "CH": float(ch)})
        print(f"[无标签评价] Silhouette={sil:.4f}, DBI={dbi:.4f}, CH={ch:.4f}")

    elapsed = time.time() - t0
    print(f"Time: {elapsed:.3f}s")
    print("Final cluster count:", len(merged_clusters))

    # 可视化
    try:
        plt.figure(figsize=(6, 5),dpi=100)
        colors = plt.cm.tab20(np.linspace(0, 1, len(merged_clusters)))
        for k, cluster in enumerate(merged_clusters):
            pts = X[list(cluster)]
            plt.scatter(pts[:, 0], pts[:, 1], s=15, color=colors[k], alpha=0.7, label=f"C{k}")
        plt.title(f"GB-AIC: {data_file}", fontsize=18)
        plt.axis("equal")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("[Warn] Plotting skipped:", e)

    return results


# ============== 批量：分裂准则消融实验 ==============
def run_ablation_split(data_file: str = "2d-10c.xlsx"):
    rules = ["BIC", "AIC", "NO_TEST", "BIC_DELTA"]
    all_rows = []

    print("\n================ 消融实验：分裂准则对比 ================")
    for rule in rules:
        CFG["split_rule"] = rule
        print(f"\n▶ 模式: {rule}  (Δ={CFG['bic_delta'] if rule=='BIC_DELTA' else 0})")
        res = run(data_file)
        row = {"Rule": rule, **res}
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    if CFG.get("save_results", True):
        out_csv = "ablation_split_results.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n✅ 结果已保存到 {out_csv}")
    print("\n================ 结束 =================\n")
    return df


# ============== 主入口 ==============
if __name__ == "__main__":
    # 默认针对你提供的数据进行四模式消融
    _df = run_ablation_split("Dataset3.xlsx")
    print(_df)