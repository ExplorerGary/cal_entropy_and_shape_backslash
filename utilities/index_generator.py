# index_generator.py
'''
目标：
保持范围够大（因为grad更新可能需要大grad），然后对范围内的东西实行不均衡分桶。
比如很小的grad卡在一个位点，其实可以放在一个桶里。
就好比教授说的0.0001-0.0002，然后慢慢地拉大桶的步长，容纳更多，更大可能的值。
但是最后总index表的长度不会很大。

'''

'''
实施方案：
0. 已知：
    bucket的数据服从GGD
    绝大多数数据都是集中在远小于1的地方，然后还有一堆比较大的离散值，目前的最高观测值为4e2
1. 算法使用scipy.stats.gennorm实现：
    根据shape parameter，构建CDF。
    然后将CDF均匀地分桶，实现“慢慢拉大桶的步长以容纳更多，大的值”
    每一个桶的有一个独特的index。
    然后桶的值就是其"(上确界+下确界)/2"或者是50百分位数(依情况而定)

2. 工作流程：
    得到shape parameter
    得到pdf,cdf还有ppf
    
    靠近0的部分使用scipy精细分桶：
    划出一个最高位（目前暂定为0.9999999999），然后算出对应的数值。
        -- 为啥是0.9999999999？
            -- 因为在gemma = 1.0, beta = 0.01, loc = 0.0 的时候，ppf = 0.9999999999将会返回0.22333，已超过观测到桶中数据的100百分位数的均值。认定为通过
        -- 超过这个部分的统统clip到长尾区域 （这是由backslash论文启发的）
    将精细部分划为n个桶（这个数据是可调的，目前暂定为1e10）
    
    而长尾就使用固定的step粗犷分桶
        目前的起点估计定为0.25
        step为0.25
        到1000，会有4000个bucket
        用EG encoding后
        即使精细部分花了1e6个桶，我们最高也就只有1004000个index，也就是40个bit就能编码
                               1e8个精细桶也就是54bit
        
    将桶的映射保存为一个.pt文件
    并且返回这个.pt文件的地址    
'''

import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import gennorm
import matplotlib.pyplot as plt

base_dir = os.path.dirname(__file__)
storge_path = os.path.join(base_dir,"..","data_to_use")


def index_generator(
    gemma: float,
    beta: float,
    mu: float = 0.0,
    ppf_end_point: float = 0.9999999999,
    end_point: float = 1e3,
    bucket_num: int = int(1e8),  # 精细部分桶数（可调），但GPT建议使用1e6防止爆内存
    step: float = 0.25,
    tail_start: float = 0.25,
    save_dir: str = storge_path,
    save_name: str = "ggd_index_table.pt"
) -> str:
    
    """
    构造精细 + 粗长尾混合桶，生成梯度非均匀分桶的 index 映射表

    :param gemma: GGD scale 参数（越小分布越集中）
    :param beta: GGD shape 参数（控制尖度和尾部）
    :param mu: GGD 平移参数（通常为0）
    :param ppf_end_point: 精细分桶的最大CDF阈值
    :param end_point: 长尾最大值，超出此值统一clip
    :param bucket_num: 精细部分分桶数量（太大可能爆内存）
    :param step: 长尾部分步长
    :param tail_start: 长尾起点，建议略大于精细区最大值
    :param save_dir: 保存路径
    :param save_name: 保存文件名
    :return: 保存路径
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. 创建 GGD 对象
    ggd = gennorm(beta = gemma, loc=mu, scale=beta)

    # 2️. 获取精细部分最大值（ppf_end_point 对应的实际数值）
    fine_max_value = ggd.ppf(ppf_end_point)
    print(f"[INFO] 精细分桶最大值 ≈ {fine_max_value:.6f}")

    # 3️. 构建 CDF 等分区间
    fine_cdf = np.linspace(0, ppf_end_point, bucket_num + 1)
    fine_bucket_centers = []
    for i in range(bucket_num):
        lower = fine_cdf[i]
        upper = fine_cdf[i + 1]
        mid = (lower + upper) / 2
        val = ggd.ppf(mid)
        fine_bucket_centers.append(val)

    fine_bucket_centers = np.array(fine_bucket_centers, dtype=np.float32)

    # 4️. 构建粗长尾部分（均匀步长从 tail_start 到 end_point）
    tail_values = np.arange(tail_start, end_point + step, step, dtype=np.float32)
    print(f"[INFO] 长尾桶数量：{len(tail_values)}")

    # 5️. 合并两个部分
    all_buckets = np.concatenate([fine_bucket_centers, tail_values])
    all_buckets = np.clip(all_buckets, a_min=None, a_max=end_point)

    # 6️. 保存为 .pt
    index_table = torch.tensor(all_buckets)
    save_path = os.path.join(save_dir, save_name)
    torch.save(index_table, save_path)

    print(f"[SUCCESS] Index table saved to: {save_path}")
    print(f"[INFO] Total number of buckets: {len(index_table)}")
    return save_path

def local_test():
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    from scipy.stats import gennorm

    # === 参数设定 ===
    gemma: float = 1.0
    beta: float = 0.01
    mu: float = 0.0
    ppf_end_point: float = 0.9999999999
    end_point: float = 1e3
    bucket_num: int = int(1e8)  # 防止爆内存
    step: float = 0.25
    tail_start: float = 0.25
    save_name = "gamma_table.pt"

    # === 生成 index 表 ===
    index_path = index_generator(
        gemma=gemma,
        beta=beta,
        mu=mu,
        ppf_end_point=ppf_end_point,
        end_point=end_point,
        bucket_num=bucket_num,
        step=step,
        tail_start=tail_start,
        save_name=save_name,
    )

    # === 加载 index 表 ===
    stuff = torch.load(index_path, map_location="cpu").numpy()
    print(f"Index table shape: {stuff.shape}")
    print(f"前10个桶: {stuff[:10]}")
    print(f"最后10个桶: {stuff[-10:]}")

    # === 创建 GGD 分布对象 ===
    ggd = gennorm(beta = gemma, loc=mu, scale=beta)

    # === 绘制参考分布图 ===
    x_vals = np.linspace(0, 1.0, 1000)
    pdf_vals = ggd.pdf(x_vals)
    cdf_vals = ggd.cdf(x_vals)
    ppf_vals = ggd.ppf(np.linspace(1e-10, 0.9999999999, 1000))  # 避免log爆炸

    plt.figure(figsize=(16, 8))

    # PDF
    plt.subplot(2, 2, 1)
    plt.plot(x_vals, pdf_vals, label="PDF", color="blue")
    plt.title("GGD PDF")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("Density")

    # CDF
    plt.subplot(2, 2, 2)
    plt.plot(x_vals, cdf_vals, label="CDF", color="green")
    plt.title("GGD CDF")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("Cumulative Prob.")

    # PPF
    plt.subplot(2, 2, 3)
    probs = np.linspace(1e-10, 0.9999999999, 1000)
    plt.plot(probs, ppf_vals, label="PPF", color="purple")
    plt.title("GGD PPF (Inverse CDF)")
    plt.grid(True)
    plt.xlabel("Probability")
    plt.ylabel("x")

    # 桶分布
    plt.subplot(2, 2, 4)
    plt.plot(stuff, label="Index bucket values", color="red")
    plt.yscale("log")
    plt.title("Bucket Center Values (log scale)")
    plt.grid(True)
    plt.xlabel("Bucket Index")
    plt.ylabel("Value (log)")

    plt.tight_layout()
    plt.suptitle("GGD Bucket Mapping Summary", fontsize=16, y=1.03)
    plt.show()

local_test()

# # from scipy.stats import norm

# # # 计算标准正态分布下 P(X <= 1.0)
# # print(norm.cdf(5))  # 输出约为 0.999999999999
# # print(norm.cdf(10)) # 直接报1.0
# # print(norm.cdf(0.0))  # 输出约为 0.5000

# # 1. 构造 GGD
# ggd = gennorm(beta=1.0, loc=0.0, scale=0.01)

# # 2. 选择一个比较安全的右尾概率（例如 0.9999）
# target_cdf_tail = 0.9999999999
# print(ggd.cdf(0.22333))
# # 3. 用 PPF 反推出这个概率对应的值
# endpoint = ggd.ppf(target_cdf_tail)
# print(f"合理的 endpoint（包含99.99%概率质量）是：{endpoint:.5f}")