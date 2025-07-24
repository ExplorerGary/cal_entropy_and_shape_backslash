# cal_entropy:
"""
熵定义为 -sum(p.*log2(p))，其中 p 包含从 imhist 返回的归一化直方图计数。

现已更改为对于bucket中每一个数据值进行运算

"""
import numpy as np
import torch
import os
from .utilities import read_pt, quantlization, scan_pt
import time

def cal_entropy(pt_path:str,
                fp64_enable:bool = False, 
                pure_data_enable:bool = False, 
                scaling = int(1e6),
                abs_enabled:bool = False) -> float:
    """
    计算熵
    :param pt_path: 输入pt文件的路径
    :fp64_enable: 启用fp64
    :param fp64_enable: bool - 是否启用64位浮点数计算模式(默认False)
    :param pure_data_enable: bool - 是否处理纯数据模式(默认False)
    :param scaling: int - 数据缩放因子(默认1e6)，用于调整数据范围以便计算
    :param abs_enabled: bool - 判断是否需要使用torch.abs返回
    
    :return: dict: 名称和对应的熵值(单位为bit)
    """
    try:
        name = os.path.basename(pt_path)
        pt_array = torch.abs(read_pt(pt_path)) if abs_enabled else read_pt(pt_path)
        quantized = pt_array if pure_data_enable else quantlization(pt_array=pt_array,
                                                                    scaling = scaling, 
                                                                    fp64_enable = fp64_enable,
                                                                    debug = True)
        # 按照(量化)后的每一个数据值进行计算

        # 统计每个唯一值的出现次数
        values, counts = np.unique(quantized, return_counts=True)

        ### debugging info ###
        # print(f"how long the bucket is = {len(quantized)}\nhow many bin(unique) nums = values}")
        ######################

        
        # 计算频率（概率）
        probs = counts / counts.sum()
    
        # 计算熵（香农熵，单位为比特）
        entropy = -np.sum(probs * np.log2(probs))
        
        return {
            "name":name, 
            "entropy":entropy
            }
    except Exception as e:
        print(e)
        return {
            "name":None, 
            "entropy":None
            }

def local_test():
    """
    本地测试函数
    """
    base_dir = "D:\\NYU_Files\\2025 SPRING\\Summer_Research\\新\\PYTHON\\QWEN\\dummy_files\\"
    avail_pt = scan_pt(base_dir=base_dir)
    print(f"found: {len(avail_pt)}")
    for pt_path in avail_pt:
        start = time.time()
        result = cal_entropy(pt_path,fp64_enable=True,pure_data_enable=True,scaling=int(1e8))
        end = time.time()
        print(f"Entropy of {result['name']}: {result['entropy']}\n it takes: {end - start}\n\n")
    
    
# local_test()


# def quantization(pt_array:np.ndarray,scaling = 2 ** 8) -> np.ndarray:
#     """
#     对输入的pt_array进行量化处理：
#     先将pt_array乘以2的8次方，四舍五入后再除以2的8次方，返回量化后的数组。
#     :param pt_array: 输入的numpy数组
#     :param scaling: 量化的缩放因子，默认为2的8次方
#     :return: 量化后的numpy数组
#     """
#     if not isinstance(pt_array, np.ndarray):
#         raise TypeError("pt_array must be a numpy array")
#     quantized = np.round(pt_array * scaling) / scaling
#     return quantized
    
#     pass


def cal_entropy_bucket(pt_path:str,bins:int=256) -> float:
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