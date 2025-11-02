import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def reduce_and_save(input_csv: str, output_xlsx: str = "reduced_output.xlsx", n_components: int = 2):
    """
    从CSV读取数据，对特征部分进行PCA降维，保留类别列，并导出为xlsx，同时可视化结果
    :param input_csv: 输入CSV文件路径
    :param output_xlsx: 输出Excel文件路径
    :param n_components: 降维后的维度数
    """
    # 读取CSV
    df = pd.read_csv(input_csv)
    data = df.values
    X, y = data[:, :-1], data[:, -1]   # 前面是特征，最后一列是类别

    # PCA降维
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)

    # 合并结果
    df_out = pd.DataFrame(X_reduced, columns=[f"PC{i+1}" for i in range(n_components)])
    df_out["label"] = y

    # 保存为xlsx
    df_out.to_excel(output_xlsx, index=False)
    print(f"降维完成，结果已保存到 {output_xlsx}")

    # ===== 可视化 =====
    if n_components == 2:
        plt.figure(figsize=(7, 6))
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="tab10", s=20, alpha=0.7)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA 2D Visualization")
        plt.colorbar(scatter, label="Class Label")
        plt.show()


if __name__ == "__main__":
    # 使用上传的csv文件
    reduce_and_save("liver.csv", "liver.xlsx")
