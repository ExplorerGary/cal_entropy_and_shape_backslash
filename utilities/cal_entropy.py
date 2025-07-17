# cal_entropy:
"""
熵 (来自matlab的定义：https://ww2.mathworks.cn/help/images/ref/entropy.html)

熵定义为 -sum(p.*log2(p))，其中 p 包含从 imhist 返回的归一化直方图计数。

默认情况下，entropy 对逻辑数组使用两个 bin，对 uint8、uint16 或 double 数组使用 256 个 bin。为计算直方图计数，entropy 将除 logical 以外的任何数据类型转换为 uint8，以使像素值离散并直接对应于 bin 值。

"""
import numpy as np
import torch
import os
from utilities import read_pt
def cal_entropy(pt_path:str,bins:int=256) -> float:
    """
    计算熵
    :param pt_path: 输入pt文件的路径
    :param bins: 直方图的bin数，默认为256
    :return: dict: 名称和对应的熵值
    """
    try:
        name = os.path.basename(pt_path)
        pt_array = read_pt(pt_path)
        if not isinstance(pt_array, np.ndarray):
            raise TypeError("pt_array must be a numpy array")
        
        hist, _ = np.histogram(pt_array, bins=bins, density=False)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        entropy = -np.sum(prob * np.log2(prob))
        
        return {
            "name":name, 
            "entorpy":entropy
            }
    except Exception as e:
        print(e)
        return {
            "name":None, 
            "entorpy":None
            }

def cal_entropy_ggd(pt_path:str,gemma,mu) -> float:
    """
    利用计算熵
    :param pt_path: 输入pt文件的路径
    :param gemma,mu: 直方图的bin数，默认为256
    :return: dict: 名称和对应的熵值
    """
    try:
        name = os.path.basename(pt_path)
        entropy = 0
        
        return {
            "name":name, 
            "entorpy_ggd":entropy
            }
    except Exception as e:
        print(e)
        return {
            "name":None, 
            "entorpy":None
            }




def local_test():
    """
    本地测试函数
    """
    pt_path = "D:\\NYU_Files\\2025 SPRING\\Summer_Research\\新\\PYTHON\\QWEN\\dummy_files\\R_1_E_0_S_9_B_8.pt"  # 替换为实际的.pt文件路径
    result = cal_entropy(pt_path)
    print(f"Entropy of {result['name']}: {result['entorpy']}")
    
    
    
# local_test()


