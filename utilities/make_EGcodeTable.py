# make_EGcodeTable
import os
import torch
import numpy as np
import pickle
from multiprocessing import Pool
try:
    from EG_encoding import ExpGolombEncoding
except:
    from .EG_encoding import ExpGolombEncoding

base_dir = os.path.dirname(__file__)
save_dir = os.path.join(base_dir,"..","data_to_use","EG_code_table")
os.makedirs(save_dir,exist_ok=True)

possible_k_table = [0,1,2,3,4,5]

### 注意，the_range是一个可以调整的超参数，决定EG_table_k的总长度
the_range = 200000
the_range *= 2 # the_range指的是一边的长度，考虑到负数，实际上应该要乘以2
to_encode = [i for i in range(the_range + 1)]




def make_EGTable_with_k(k:int = 0):
    '''
    给定参数 k，生成对应的 EG 编码查表，并保存为 .pkl 文件方便日后读入
    保存格式：EG_table_{k}.pkl
    '''
    print(f"正在生成 EG 编码表：k = {k}")
    save_file_path = os.path.join(save_dir,f"EG_table_{k}.pkl")
    EG = ExpGolombEncoding(k=k)
    codes = EG.encode(to_encode)
    
    # 存为未压缩 pickle 文件
    with open(save_file_path, "wb") as f:
        pickle.dump(codes, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ 编码表已保存：{save_file_path}（共 {len(codes)} 项）")
    
    return save_file_path

# if __name__ == "__main__":
#     # 多进程并行加速生成多个 k 的查表
#     print("🚀 正在并行生成所有 EG 编码表...")
#     with Pool(len(possible_k_table)) as pool:
#         pool.map(make_EGTable_with_k, possible_k_table)
#     print("🎉 所有编码表生成完成！")
    
def local_test():
    to_encode = [i for i in range(-200,201)]
    EG = ExpGolombEncoding(k=0)
    codes = EG.encode(to_encode)
    print(codes)

