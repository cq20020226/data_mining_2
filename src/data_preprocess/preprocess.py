import pandas as pd
import json
import os
import time
from glob import glob
import matplotlib
matplotlib.use('Agg')  # 关闭图形后端，适合远程环境

# 路径设置
dataset_dir = '/data/DM_work/dataset/30G_data_new'
product_info_path = os.path.join(dataset_dir, 'product_catalog.json')
output_dir = '/data/DM_work/output/data_preprocess_output'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'purchase_history_final.jsonl')

# 商品信息字典（id -> {category, price}）
with open(product_info_path, 'r', encoding='utf-8') as f:
    product_info = json.load(f)

id_map = {
    prod["id"]: {
        "category": prod["category"],
        "price": prod["price"]
    } for prod in product_info["products"]
}

# 小类 -> 大类 映射表
category_map = {
    "智能手机": "电子产品", "笔记本电脑": "电子产品", "平板电脑": "电子产品", "智能手表": "电子产品",
    "耳机": "电子产品", "音响": "电子产品", "相机": "电子产品", "摄像机": "电子产品", "游戏机": "电子产品",
    "上衣": "服装", "裤子": "服装", "裙子": "服装", "内衣": "服装", "鞋子": "服装",
    "帽子": "服装", "手套": "服装", "围巾": "服装", "外套": "服装",
    "零食": "食品", "饮料": "食品", "调味品": "食品", "米面": "食品",
    "水产": "食品", "肉类": "食品", "蛋奶": "食品", "水果": "食品", "蔬菜": "食品",
    "家具": "家居", "床上用品": "家居", "厨具": "家居", "卫浴用品": "家居",
    "文具": "办公", "办公用品": "办公",
    "健身器材": "运动户外", "户外装备": "运动户外",
    "玩具": "玩具", "模型": "玩具", "益智玩具": "玩具",
    "婴儿用品": "母婴", "儿童课外读物": "母婴",
    "车载电子": "汽车用品", "汽车装饰": "汽车用品"
}

# 查找 parquet 文件
start_time = time.time()
parquet_files = glob(os.path.join(dataset_dir, '*.parquet'))
if not parquet_files:
    print("❌ 未找到 Parquet 文件")
    exit()
print(f"📦 共找到 {len(parquet_files)} 个 Parquet 文件")

# 打开输出流
record_count = 0
with open(output_file, 'w', encoding='utf-8') as f_out:
    for idx, file in enumerate(parquet_files):
        print(f"\n📂 正在处理文件 {idx + 1}/{len(parquet_files)}: {file}")
        try:
            df = pd.read_parquet(file, engine='pyarrow')
        except Exception as e:
            print(f"⚠️ 读取失败: {e}")
            continue

        if 'purchase_history' not in df.columns:
            print("⚠️ 跳过：字段 'purchase_history' 不存在")
            continue

        for raw in df['purchase_history']:
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                print("⚠️ 跳过无法解析的记录")
                continue

            # 删除无效字段
            record.pop("categories", None)
            record.pop("avg_price", None)

            # 补充 items 信息
            new_items = []
            for item in record.get("items", []):
                prod_id = item.get("id")
                if prod_id in id_map:
                    category = id_map[prod_id]["category"]
                    new_items.append({
                        "id": prod_id,
                        "category": category,
                        "price": id_map[prod_id]["price"],
                        "major_category": category_map.get(category, "其他")
                    })
                else:
                    print(f"⚠️ 商品ID {prod_id} 未找到，跳过")

            # 更新记录
            record["items"] = new_items
            record["item_count"] = len(new_items)

            # 写入 jsonl
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            record_count += 1
            if record_count % 100000 == 0:
                print(f"✅ 已写入记录: {record_count:,}")

elapsed = time.time() - start_time
print(f"\n🎉 处理完成，记录数：{record_count:,}")
print(f"📁 输出文件：{output_file}")
print(f"⏱️ 总用时：{elapsed:.2f} 秒")
