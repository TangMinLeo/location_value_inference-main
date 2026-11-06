import pandas as pd

# 读取CSV文件
csv_file = './data/1_ods_bd_poi.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(csv_file)

# 提取相关列并将其存储到一个set中
categories = set(df['first_category_name'].dropna()) | \
             set(df['second_category_name'].dropna()) | \
             set(df['third_category_name'].dropna())

# 将set中的数据转换为DataFrame并保存到Excel文件
categories_df = pd.DataFrame(categories, columns=['Category'])

# 将数据保存为Excel文件
excel_file = './data/2_address.xlsx'  # 结果保存的Excel文件名
categories_df.to_excel(excel_file, index=False, engine='openpyxl')

print(f"数据已保存到 {excel_file}")
