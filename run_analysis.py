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
    "001_ENTROPY_RESULTS_PROCESSPOOL_a1.csv",
    "001_ENTROPY_RESULTS_PROCESSPOOL_b1.csv",
    "001_ENTROPY_RESULTS_PROCESSPOOL_c1.csv",
    "001_ENTROPY_RESULTS_PROCESSPOOL_d1.csv",
    "001_ENTROPY_RESULTS_PROCESSPOOL_a2.csv",
    "001_ENTROPY_RESULTS_PROCESSPOOL_b2.csv",
    "001_ENTROPY_RESULTS_PROCESSPOOL_c2.csv",
    "001_ENTROPY_RESULTS_PROCESSPOOL_d2.csv",
]
size_files = [
    "003_SIZE_RESULTS_PROCESSPOOL_b.csv",
    "003_SIZE_RESULTS_PROCESSPOOL_c.csv"
]


col_dict = {
    1:entropy_files,
    3:size_files,
}
col_name_dict = {
    1:"entropy",
    3:"size",
}



def read_csv(csv_path):
    global data_dir
    # 如果是文件名（不包含/），拼上 data_dir 路径
    if "/" not in csv_path:
        csv_path = os.path.join(data_dir, csv_path)
    df = pd.read_csv(csv_path)
    return df

def hist(df: pd.DataFrame, bins: int = 256, title: str = "", suffix = ""):
    print(f"got suffix: {suffix}")
    for col in df.columns:
        if "entropy" in df.columns:
            entropy_legend = "蓝色是原始数据，橘色是经过量化了的数据"
        data = df[col]

        # 跳过非数值列
        if data.dtype == object:
            continue

        info = f"""{col}
{suffix}
"""  # 用作 legend
        print(info)
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
    if THING_TO_WORK_ON == 1:
        for csv_file in csv_files[:4]:
            suffix = csv_file.split("_")[-1].replace(".csv", "")  # 提取类似 a1、b1、c1
            print(suffix)
            try:
                print(f"abs_disabled, handling {csv_file}...")
                df = read_csv(csv_file)
                hist(df, bins=256, title=title,suffix=suffix)
            except:
                print(f"unable to load{csv_file}, continuing...")
                continue
        # plt.close()
        for csv_file in csv_files[4:-1]:
            try:
                suffix = csv_file.split("_")[-1].replace(".csv", "")  # 提取类似 a1、b1、c1
                print(f"abs_enabled, handling {csv_file}...")
                df = read_csv(csv_file)
                hist(df, bins=256, title=str(title)+"_compare_with_abs_enabled",suffix=suffix)
            except:
                print(f"unable to load{csv_file}, continuing...")
                continue
    
    elif THING_TO_WORK_ON == 3:
        for csv_file in csv_files:
            df = read_csv(csv_file)
            avg_bit_per_entry = df["avg_bit_per_entry"]
            time_used = df["time_used"]

            # 描述信息
            info1 = f"avg_bit_per_entry\n{avg_bit_per_entry.describe()}"
            info2 = f"time_used\n{time_used.describe()}"

            # 创建一个图像，包含两个子图
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1行2列

            # 子图1: avg_bit_per_entry
            axes[0].hist(avg_bit_per_entry, bins=256, alpha=0.6, histtype='stepfilled')
            axes[0].set_title("avg_bit_per_entry")
            axes[0].legend([info1], fontsize=8)
            axes[0].grid(True)

            # 子图2: time_used
            axes[1].hist(time_used, bins=256, alpha=0.6, histtype='stepfilled')
            axes[1].set_title("time_used")
            axes[1].legend([info2], fontsize=8)
            axes[1].grid(True)

            # 保存图像
            plt.tight_layout()
            save_path = os.path.join(data_dir, f"{title}_summary_{os.path.basename(csv_file).replace('.csv','')}.png")
            plt.savefig(save_path)
            plt.close()
