import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os
import matplotlib
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False


# 路径设置
DATA_PATH = "/data/data_mining_2/output/data_preprocess_output/purchase_history_final.jsonl"  # 替换为你的文件名
OUTPUT_DIR = "/data/DM_work/output/analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载数据
# def load_data(path):
#     with open(path, 'r', encoding='utf-8') as f:
#         return json.load(f)
# def load_data(path):
#     return load_jsonl_as_list(path)
# def load_jsonl_as_list(path):
#     """将 .jsonl 文件转换为 JSON 列表格式"""
#     data = []
#     with open(path, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 try:
#                     data.append(json.loads(line))
#                 except json.JSONDecodeError:
#                     print(f"⚠️ 跳过无法解析的行: {line}")
#     return data
def load_jsonl_as_list(path, limit=None):
    """将 .jsonl 文件转换为 JSON 列表格式（可限制最多读取多少行）"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"⚠️ 跳过无法解析的行: {line}")
    return data

# 提取每个订单中的 major_category 集合
def extract_major_categories(orders):
    return [list(set(item['major_category'] for item in order['items'])) for order in orders]


def task1(data):
    # 确保输出目录存在
    task_output_dir = os.path.join(OUTPUT_DIR, "task1")
    os.makedirs(task_output_dir, exist_ok=True)


    # 提取订单中所有商品的大类
    transactions = extract_major_categories(data)

    # One-hot 编码
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # 1. 挖掘频繁项集（支持度 ≥ 0.02）
    frequent_itemsets = apriori(df, min_support=0.02, use_colnames=True)
    frequent_itemsets.to_csv(f"{task_output_dir}/task1_frequent_itemsets.csv", index=False)

    # 2. 生成关联规则（置信度 ≥ 0.5）
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    rules.to_csv(f"{task_output_dir}/task1_rules.csv", index=False)

    # 3. 特别筛选出与“电子产品”相关的规则
    electronics_rules = rules[
        rules['antecedents'].apply(lambda x: '电子产品' in x) |
        rules['consequents'].apply(lambda x: '电子产品' in x)
    ]
    electronics_rules.to_csv(f"{task_output_dir}/task1_electronics_rules.csv", index=False)

    print("✅ 任务1完成：频繁项集、关联规则、电子产品相关规则已保存")
    return frequent_itemsets, rules, electronics_rules

def task2(data):
    # if not os.path.exists(OUTPUT_DIR):
    #     os.makedirs(OUTPUT_DIR)
    task_output_dir = os.path.join(OUTPUT_DIR, "task2")
    os.makedirs(task_output_dir, exist_ok=True)

    high_value_data = []
    payment_counter = Counter()

    # 1️⃣ 筛选含有单价 ≥ 5000 的商品的订单
    for order in data:
        if any(item['price'] >= 5000 for item in order['items']):
            payment_method = order['payment_method']
            payment_counter[payment_method] += 1

            categories = list(set(item['major_category'] for item in order['items']))
            transaction = categories + [payment_method]
            high_value_data.append(transaction)

    # 2️⃣ 保存首选支付方式统计
    payment_df = pd.DataFrame(payment_counter.items(), columns=["payment_method", "count"])
    payment_df = payment_df.sort_values(by="count", ascending=False)
    payment_df.to_csv(f"{task_output_dir}/task2_payment_method_stats.csv", index=False)

    print("💳 高价商品订单中支付方式统计已保存：task2_payment_method_stats.csv")

    # 3️⃣ 转换交易格式
    te = TransactionEncoder()
    te_ary = te.fit(high_value_data).transform(high_value_data)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # 4️⃣ 挖掘频繁项集并保存
    frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
    frequent_itemsets.to_csv(f"{task_output_dir}/task2_frequent_itemsets.csv", index=False)
    print("📦 频繁项集已保存：task2_frequent_itemsets.csv")

    # 5️⃣ 生成关联规则并保存
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
    rules.to_csv(f"{task_output_dir}/task2_rules.csv", index=False)
    print("✅ 关联规则已保存：task2_rules.csv")

    return rules


def task3(data):
    task_output_dir = os.path.join(OUTPUT_DIR, "task3")
    os.makedirs(task_output_dir, exist_ok=True)


    # === 构建基础 DataFrame ===
    df = pd.DataFrame(data)
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['month'] = df['purchase_date'].dt.month
    df['quarter'] = df['purchase_date'].dt.quarter
    df['weekday'] = df['purchase_date'].dt.dayofweek  # 0=Monday, 6=Sunday

    # ✅ 任务3.1：季节性分析（季度 & 周几）
    # —— 季度分析
    quarter_counts = defaultdict(Counter)
    for _, row in df.iterrows():
        quarter = row['quarter']
        for item in row['items']:
            quarter_counts[quarter][item['major_category']] += 1
    pd.DataFrame(quarter_counts).fillna(0).astype(int).to_csv(f"{task_output_dir}/task3_quarterly_category_counts.csv")
    print("完成3.1")
    # —— 每周周期分析
    weekday_counts = defaultdict(Counter)
    for _, row in df.iterrows():
        weekday = row['weekday']
        for item in row['items']:
            weekday_counts[weekday][item['major_category']] += 1
    pd.DataFrame(weekday_counts).fillna(0).astype(int).to_csv(f"{task_output_dir}/task3_weekday_category_counts.csv")
    print("完成3.2")
    # ✅ 任务3.2：商品类别月度趋势图（已完成）
    monthly_counts = []
    for month in range(1, 13):
        items = []
        for order in df[df['month'] == month]['items']:
            items.extend(item['major_category'] for item in order)
        count = Counter(items)
        monthly_counts.append(count)

    months = list(range(1, 13))
    plt.figure(figsize=(12, 6))
    for category in set(cat for counter in monthly_counts for cat in counter):
        plt.plot(months, [month_count.get(category, 0) for month_count in monthly_counts], label=category)

    plt.xlabel("月份")
    plt.ylabel("购买数量")
    plt.title("商品大类的月度购买趋势")
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.savefig(f"{task_output_dir}/task3_monthly_trends.png")
    plt.close()
    print("完成3.3")
    # ✅ 任务3.3：时序模式分析（先 A 后 B 的大类对）
    # 逻辑：按用户的购买时间先后排序，收集每个用户购买大类的顺序
    # 假设每条记录来自一个用户（如果有 user_id，可以进一步细化）
    # 注意：这里只能做简单顺序，非严格间隔关系（如买了A，未来某时买B）

    sequence_pairs = Counter()
    user_sequences = []

    df_sorted = df.sort_values('purchase_date')
    for _, row in df_sorted.iterrows():
        # 获取该订单的大类集合（去重）
        categories = list(set(item['major_category'] for item in row['items']))
        user_sequences.append(categories)

    # 建立大类对的转移（先A后B）
    for i in range(len(user_sequences) - 1):
        current = user_sequences[i]
        next_ = user_sequences[i + 1]
        for a in current:
            for b in next_:
                if a != b:
                    sequence_pairs[(a, b)] += 1

    seq_df = pd.DataFrame([
        {"from_category": a, "to_category": b, "count": c}
        for (a, b), c in sequence_pairs.items()
    ])
    seq_df.sort_values(by="count", ascending=False).to_csv(f"{task_output_dir}/task3_sequential_category_pairs.csv", index=False)
    print("完成3.4")
    print("✅ 任务3完成：")
    print("  📊 月度趋势图已保存为 task3_monthly_trends.png")
    print("  📁 季度分类统计已保存为 task3_quarterly_category_counts.csv")
    print("  📁 每周分类统计已保存为 task3_weekday_category_counts.csv")
    print("  🔁 时序模式已保存为 task3_sequential_category_pairs.csv")


def task4(data):
    task_output_dir = os.path.join(OUTPUT_DIR, "task4")
    os.makedirs(task_output_dir, exist_ok=True)


    refund_data = []

    # 1️⃣ 提取退款订单中的商品大类
    for order in data:
        if order['payment_status'] in ['已退款', '部分退款']:
            categories = list(set(item['major_category'] for item in order['items']))
            refund_data.append(categories)

    if refund_data:
        # 2️⃣ 转换成布尔型编码
        te = TransactionEncoder()
        te_ary = te.fit(refund_data).transform(refund_data)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        # 3️⃣ 频繁项集挖掘
        frequent_itemsets = apriori(df, min_support=0.005, use_colnames=True)
        frequent_itemsets.to_csv(f"{task_output_dir}/task4_frequent_itemsets.csv", index=False)
        print("📦 task4_frequent_itemsets.csv 已保存")

        # 4️⃣ 关联规则挖掘
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)
        rules.to_csv(f"{task_output_dir}/task4_rules.csv", index=False)
        print("✅ task4_rules.csv 已保存：包含退款相关商品组合模式")

        return rules
    else:
        # 没有退款订单的情况
        with open(f"{task_output_dir}/task4_rules.csv", "w", encoding='utf-8') as f:
            f.write("无退款订单数据")
        with open(f"{task_output_dir}/task4_frequent_itemsets.csv", "w", encoding='utf-8') as f:
            f.write("无退款订单数据")
        print("⚠️ 任务4：没有退款订单，未生成频繁项集与规则")
        return None

# 主函数
def main():
    print("📥 正在加载数据...")
    data = load_jsonl_as_list(DATA_PATH)

    print("🚀 执行任务1...")
    task1(data)

    print("🚀 执行任务2...")
    task2(data)

    print("🚀 执行任务3...")
    task3(data)

    print("🚀 执行任务4...")
    task4(data)



    # print("✅ 所有任务完成，结果保存在 results_major 目录")

if __name__ == "__main__":
    main()
