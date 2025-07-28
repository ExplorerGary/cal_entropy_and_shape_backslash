# EG_preproces_deprocess.py
'''
将一个torch.tensor转化为可供EG Encoding所使用的整数数列/解码为经过量化的数据
'''

import torch
import numpy as np
import os


# if torch.cuda.is_available():
#     loc = torch.device("cuda")
# else:
#     loc = torch.device("cpu")

base_dir = os.path.dirname(__file__)
table_path = os.path.join(base_dir, "..", "data_to_use", "ggd_index_table.pt")  # table存储的位置
table = torch.load(table_path, map_location="cpu")

index2value = table["index2value"].to("cpu") # it's a dict.
boundaries = table["boundaries"].to("cpu")
print("table loaded...")




def preprocess(pt_path:str = None,pt_array:torch.Tensor = None):
    '''
    读入一个原始的tensor
        记录其符号
        归类如桶获得index
        返回带符号的index供改修过了的EG encoding使用
    
    '''
    global boundaries
    
    if pt_array is None:
        try:
            pt_array = torch.load(pt_path,map_location="cpu")
        except:
            raise ValueError("Unable to process: No Tensor to Process!!!!!")
        
    
    sign = torch.sign(pt_array).to(torch.int8) # get the sign   
    index = torch.bucketize(pt_array, boundaries, right=False)
    
    signed_index = sign * index
    
    return signed_index

def deprocess(pt_path:str = None,pt_array:torch.tensor = None):
    '''
    读入一个原始的tensor
        记录其符号
        对其取abs获取原始index
        利用index获取原先的数据
        乘上符号
        返回带符号的解码数据供日后all_reduce等梯度更新使用，继续训练
    
    '''
    if pt_path:
        pt_array = torch.load(pt_path,map_location="cpu")
    
    if not pt_array:
        raise ValueError("Unable to process: No Tensor to Process!!!!!")
    
    sign = torch.sign(pt_array).to(torch.int8) # get the sign 
    index = torch.abs(pt_array)
    data = index2value[index] * sign
    
    return data

    
    
