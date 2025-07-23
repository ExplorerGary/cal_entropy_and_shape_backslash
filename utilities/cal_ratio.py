# run_cal_ratio.py

import os
import numpy as np
import torch
from .utilities import read_pt,to_int,quantlization
from .EG_encoding import ExpGolombEncoding

def preprocess_fuct(pt_path:str,scale:float=1e6):
    '''
    阅读一个pt文件，进行预处理进行量化
    预备后续处理
    
    返回 处理过的array，理论和由os.path.getsize得到的字节数
    
    '''
    quantized  =  quantlization(read_pt(pt_path=pt_path),scale=scale)
    byte_theory = 2 * len(quantized)
    byte_os = os.path.getsize(pt_path)
    
    return quantized ,byte_theory,byte_os

def preprocess(pt_path:str,scale:float=1e6):
    try:
        return preprocess_fuct(pt_path=pt_path,scale=scale)
    except Exception as e:
        print(e)
        return None
    
    
    
def cal_ratio_fuct(pt_path:str,k:int=0,scale:float=1e6):
    '''
    args: k = 0, 传入ExpGolombEncoding的k
    对一个pt文件进行编码
    计算其压缩比率
    删除原压缩文件
    返回各个答案
    '''
    name = os.path.basename(pt_path)
    quantized, byte_theory, byte_os  = preprocess(pt_path=pt_path,scale = scale)
    
    
    ### 对 quantized 进行处理 ###
    '''
    1. 构建表格
    2. 根据表格查找index (量化步长作为offset?)
    3. 
    
    '''
    
    
    ############################
    
    
    
    
    ### 不处理的 EG Encoding ###
    
    EG = ExpGolombEncoding(k=k)
    
    # 改进：使用临时变量，用完后即使删去
    # ✅ Bitstream 临时变量，及时释放
    encoded_bitstream = EG.streamEncode(quantized)
    byte_encoded = len(encoded_bitstream) // 8
    del encoded_bitstream  # 🔥 强制释放
    
    #############################
    
    ratio_theory = (byte_theory-byte_encoded)/byte_theory
    ratio_os = (byte_os-byte_encoded)/byte_os
    
    
    return {"name":name,
            "byte_theory":byte_theory,
            "byte_os":byte_os,
            "byte_encoded":byte_encoded,
            "ratio_theory":ratio_theory,
            "ratio_os":ratio_os
            }

def cal_ratio(pt_path:str,k:int=0,scale:float=1e6):
    try:
        return cal_ratio_fuct(pt_path=pt_path,k=k,scale=scale)
    except Exception as e:
        print(e)
        return None