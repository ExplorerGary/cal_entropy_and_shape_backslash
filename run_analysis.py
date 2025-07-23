import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# 获取当前脚本路径
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "data_obtained")

# 要读取的 CSV 文件名
entropy_files = [
    "001_ENTROPY_RESULTS_PROCESSPOOL_a.csv",
    "001_ENTROPY_RESULTS_PROCESSPOOL_b.csv"
]
col_dict = {
    1:entropy_files
}
col_name_dict = {
    1:"entropy"
}
def read_csv(csv_path):
    global data_dir
    # 如果是文件名（不包含/），拼上 data_dir 路径
    if "/" not in csv_path:
        csv_path = os.path.join(data_dir, csv_path)
    df = pd.read_csv(csv_path)
    return df

def hist(df: pd.DataFrame, bins: int = 256, title: str = ""):
    for col in df.columns:
        if "entropy" in df.columns:
            entropy_legend = "蓝色是原始数据，橘色是经过量化了的数据"
        data = df[col]

        # 跳过非数值列
        if data.dtype == object:
            continue

        info = f"""{col}
{data.describe()}
"""  # 用作 legend

        # 画直方图
        plt.hist(data, bins=bins, alpha=0.6, label=info, histtype='stepfilled')

        plt.title(title)
        plt.legend(fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir,f"{title}.png"))
        # plt.show()

# 执行主程序
if __name__ == "__main__":
    GREETINGS = '''
[SUPER TERMINAL ANALYSIS]
000a,b,c.txt：debug文件
001.csv: entropy
002.csv: shape parameter
003.csv: 原始bucket的理论和实际大小(单位为bit)
004.csv：经过EG压缩后的bucket的大小（单位为bit）和平均码长（单位为biit）
zzz.csv: 经过扫盘得出的.pt文件列表，以csv格式存储，到时候会merge一下，方便找出失败的格式
'''
    print(GREETINGS)
    print()
    THING_TO_WORK_ON = int(input("输入对应数字来执行分析\n\n>>>\t"))
    csv_files = col_dict[THING_TO_WORK_ON]
    title = col_name_dict[THING_TO_WORK_ON]
    for csv_file in csv_files:
        df = read_csv(csv_file)
        hist(df, bins=256, title=title)

