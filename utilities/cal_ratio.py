# run_cal_ratio.py

import os
import numpy as np
import torch
import time
try:
    from .utilities import read_pt_tensor,to_int,quantlization_pt,scan_pt
    from .EG_preprocess_deprocess import preprocess,deprocess
    from .EG_encoding import ExpGolombEncoding
except:
    from utilities import read_pt_tensor,to_int,quantlization_pt,scan_pt
    from EG_encoding import ExpGolombEncoding
    from EG_preprocess_deprocess import preprocess,deprocess 

EG = ExpGolombEncoding()
bit_per_entry_dict = {
    torch.float64: 64,  # 双精度浮点数 (double)
    torch.float32: 32,  # 单精度浮点数 (float)
    torch.float16: 16,  # 半精度浮点数 (half)
    torch.bfloat16: 16, # 脑浮点数 (bfloat16)
    torch.int64: 64,    # 64位整数 (long)
    torch.int32: 32,    # 32位整数 (int)
    torch.int16: 16,    # 16位整数 (short)
    torch.int8: 8,      # 8位整数
    torch.uint8: 8,     # 8位无符号整数
    torch.bool: 8,      # 布尔值 (通常占8位)
    torch.complex64: 64,  # 32位浮点复数 (实部+虚部各32位)
    torch.complex128: 128 # 64位浮点复数 (实部+虚部各64位)
}


def cal_ratio(pt_path,
              scaling:float= int(1e6),
              debug:bool = False
              ):
    '''
    1. 从pt_path读入一个.pt文件
    2. 对数据进行quantlization，得到被quantlized的tensor
    3. 将数据进行EG_encoding。
    4. 计算并返回:
        名称: name
        位大小: compressed_bit_size
        原始理论字节长度：original_bit_theory
        平均码长: avg_bit_per_entry
        用时: time_used
    '''
    pt_array:torch.Tensor = read_pt_tensor(pt_path=pt_path)
    the_type = pt_array.dtype
    bit_per_entry = bit_per_entry_dict[the_type]
    print(f"it's a tensor of {the_type}, every entry takes {bit_per_entry} bit")
    start = time.time()
    
    if debug:
        print("doing quantlization...")
    quantized = quantlization_pt(pt_array=pt_array,
                              scaling=scaling,
                              fp64_enable=True,
                              debug=False) # torch.Tensor
    if debug:
        print("preprocessing...")
    signed_index = preprocess(pt_path = None,
                              pt_array=quantized) # torch.Tensor
    
    if debug:
        print("encoding...")
    codes = EG.encode(signed_index,debug = debug) # -> list[str]
    
    end = time.time()
    
    #### calculating all infos ####
    name = os.path.basename(pt_path)
    compressed_bit_size = sum(len(code) for code in codes)
    original_bit_theory = len(codes) * bit_per_entry  
    avg_bit_per_entry = compressed_bit_size/len(codes)
    time_used = end - start
    
    return {
        "name" : name,
        
        "compressed_bit_size" : compressed_bit_size,
        
        "original_bit_theory" : original_bit_theory,
        
        "avg_bit_per_entry" : avg_bit_per_entry,
        
        "time_used" :time_used
    }
    
    
    
    
def local_test():
    base_dir = "D:\\NYU_Files\\2025 SPRING\\Summer_Research\\新\\PYTHON\\QWEN\\dummy_files\\"
    avail_pt = scan_pt(base_dir=base_dir)
    for pt_path in avail_pt:
        print("calculating...")
        ans = cal_ratio(pt_path,
                        scaling=int(1e6),
                        debug = True)
        for key,value in ans.items():
            print(f"""[INFO]
{key}\t\t\t\t{value}
                  """)
        
        print()



local_test()