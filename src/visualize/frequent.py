import pandas as pd
import matplotlib.pyplot as plt
import ast
import re 
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
def task1():
    def parse_itemsets(x):
    # 去除字符串开头的 "frozenset(" 和结尾的 ")"
        x = re.sub(r'^frozenset\(|\)$', '', x)
        return ', '.join(ast.literal_eval(x))
    # === 读取频繁项集 CSV 文件 ===
    frequent_itemsets_path = "/data/DM_work/output/analysis_output/task1/task1_frequent_itemsets.csv"  # 改为你的文件路径
    df = pd.read_csv(frequent_itemsets_path)

    # === 转换 itemsets 字段为字符串 ===
    df['itemsets'] = df['itemsets'].apply(parse_itemsets)

    # === 选取支持度最高的前15项 ===
    top_itemsets = df.sort_values(by='support', ascending=False).head(15)

    # === 绘制条形图 ===
    plt.figure(figsize=(10, 6))
    plt.barh(top_itemsets['itemsets'], top_itemsets['support'], color='skyblue')
    plt.xlabel("支持度 (Support)")
    plt.title("Top 15 高频商品组合")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("/data/DM_work/output/analysis_output/task1/task1_frequent_itemsets_barplot.png")  # 输出图片
  
def task2():
    def parse_itemsets(x):
    # 去除字符串开头的 "frozenset(" 和结尾的 ")"
        x = re.sub(r'^frozenset\(|\)$', '', x)
        return ', '.join(ast.literal_eval(x))
    # === 读取频繁项集 CSV 文件 ===
    frequent_itemsets_path = "/data/DM_work/output/analysis_output/task2/task2_frequent_itemsets.csv"  # 改为你的文件路径
    df = pd.read_csv(frequent_itemsets_path)

    # === 转换 itemsets 字段为字符串 ===
    df['itemsets'] = df['itemsets'].apply(parse_itemsets)

    # === 选取支持度最高的前15项 ===
    top_itemsets = df.sort_values(by='support', ascending=False).head(15)

    # === 绘制条形图 ===
    plt.figure(figsize=(10, 6))
    plt.barh(top_itemsets['itemsets'], top_itemsets['support'], color='skyblue')
    plt.xlabel("支持度 (Support)")
    plt.title("Top 15 高频商品组合")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("/data/DM_work/output/analysis_output/task2/task2_frequent_itemsets_barplot.png")  # 输出图片

def task4():
    def parse_itemsets(x):
    # 去除字符串开头的 "frozenset(" 和结尾的 ")"
        x = re.sub(r'^frozenset\(|\)$', '', x)
        return ', '.join(ast.literal_eval(x))
    # === 读取频繁项集 CSV 文件 ===
    frequent_itemsets_path = "/data/DM_work/output/analysis_output/task4/task4_frequent_itemsets.csv"  # 改为你的文件路径
    df = pd.read_csv(frequent_itemsets_path)

    # === 转换 itemsets 字段为字符串 ===
    df['itemsets'] = df['itemsets'].apply(parse_itemsets)

    # === 选取支持度最高的前15项 ===
    top_itemsets = df.sort_values(by='support', ascending=False).head(15)

    # === 绘制条形图 ===
    plt.figure(figsize=(10, 6))
    plt.barh(top_itemsets['itemsets'], top_itemsets['support'], color='skyblue')
    plt.xlabel("支持度 (Support)")
    plt.title("Top 15 高频商品组合--退款")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("/data/DM_work/output/analysis_output/task4/task4_frequent_itemsets_barplot.png")  # 输出图片

if __name__ =="__main__":
    task4()