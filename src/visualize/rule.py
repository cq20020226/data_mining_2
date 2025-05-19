import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
# 读取CSV文件（替换为你自己的文件路径）
def task2():
    file_path = "/data/DM_work/output/analysis_output/task2/task2_rules.csv"
    df = pd.read_csv(file_path)

    # 创建有向图
    G = nx.DiGraph()

    # 向图中添加边，并使用置信度作为权重
    for _, row in df.iterrows():
        antecedent = list(eval(row['antecedents']))[0]
        consequent = list(eval(row['consequents']))[0]
        confidence = row['confidence']
        G.add_edge(antecedent, consequent, weight=confidence)

    # 设置图中节点的位置
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # 绘制图形
    plt.figure(figsize=(12, 8))
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=3000,
        node_color='lightblue',
        font_size=10,
        font_weight='bold',
        arrows=True
    )

    # 添加边标签（置信度）
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Association Rules Network Graph (Edge Weight = Confidence)")
    plt.title("Association Rules Network Graph (Edge Weight = Confidence)")

def task3():
    # 读取数据
    df = pd.read_csv("/data/DM_work/output/analysis_output/task3/task3_sequential_category_pairs.csv")

    # 创建有向图
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['from_category'], row['to_category'], weight=row['count'])

    # 使用 Kamada-Kawai 布局
    pos = nx.kamada_kawai_layout(G)

    # 获取边和边权重
    edges = G.edges(data=True)
    edge_weights = [d['weight'] / 10000000 for (_, _, d) in edges]  # 缩放边宽

    # 开始绘图
    plt.figure(figsize=(18, 14))

    # ---- 1. 先画箭头（要放在节点前面）----
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color='gray',
        width=edge_weights,
        alpha=0.8,
        arrows=True,
        arrowsize=30,  # ⬅️ 加大箭头
        connectionstyle='arc3,rad=0.1'
    )

    # ---- 2. 再画节点（避免遮挡箭头）----
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1400)  # ⬅️ 节点稍微小一点
    nx.draw_networkx_labels(G, pos, font_size=12)

    # ---- 3. 画边权重标签 ----
    edge_labels = {(u, v): f"{d['weight'] // 10000}万" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    # ---- 4. 标题和美化 ----
    plt.title("购买时序", fontsize=16)
    plt.axis('off')
    plt.tight_layout()

    # ---- 5. 保存 ----
    plt.savefig("/data/DM_work/output/analysis_output/task3/category_flow_weighted.png", dpi=300)


def task4():
    file_path = "/data/DM_work/output/analysis_output/task4/task4_rules.csv"
    df = pd.read_csv(file_path)

    # 创建有向图
    G = nx.DiGraph()

    # 向图中添加边，并使用置信度作为权重
    for _, row in df.iterrows():
        antecedent = list(eval(row['antecedents']))[0]
        consequent = list(eval(row['consequents']))[0]
        confidence = row['confidence']
        G.add_edge(antecedent, consequent, weight=confidence)

    # 设置图中节点的位置
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # 绘制图形
    plt.figure(figsize=(12, 8))
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=3000,
        node_color='lightblue',
        font_size=10,
        font_weight='bold',
        arrows=True
    )

    # 添加边标签（置信度）
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Association Rules Network Graph (Edge Weight = Confidence)")
    plt.title("Association Rules Network Graph (Edge Weight = Confidence)")
# 保存图像
    plt.savefig("/data/DM_work/output/analysis_output/task4/association_rules_graph.png", dpi=300, bbox_inches='tight')
if __name__ =="__main__":
    task4()

