# run
import os
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from tqdm import tqdm
import multiprocessing
import traceback

from utilities.cal_distribution import cal_distribution,cal_distribution2
'''
def cal_distribution(pt_path:str,device=None,sample_enabled:bool=False,sample_size:int = 10000, to64 = True) -> dict:
def cal_distribution2(pt_path:str,device=None,sample_enabled:bool=False,sample_size:int = 10000,to64:bool = True) -> dict:

'''




from utilities.utilities import scan_pt
"""
    计算shape_parameters
    :param pt_path: 输入pt文件的路径
    :param filter_percentile: 过滤极值的百分位数
    :param enable_filter: 是否启用过滤极值
    :return: dict: 名称和对应的shape parameters
    {
    "name":name, 
    "gamma":GAMMA,
    "mu":MU,
    "beta":BETA
    }
        
"""

from utilities.ErrorLogger import ErrorLogger 

def main():
    base_dir = os.path.dirname(__file__)
    device = "cpu" # we shall use cpu to avoid cuda memory issues
    sample_enabled = False # sampling is not enabled by default, but it has little effect on the result and speed of calculation
    sample_size = 10000 # the size of the sample to use, default is 10000
    to64 = True # convert the tensor to float64, default is True, otherwise we can't get accurate results
    csv_path = os.path.abspath(os.path.join(base_dir, "data_obtained", "005a_shape_parameters.csv"))
    print("total cpu:", multiprocessing.cpu_count())
    max_workers=max(4,multiprocessing.cpu_count()-2)
    print(f"working on: {max_workers}")
    if not torch.cuda.is_available():
        print("local test")
        BASE_DIR = "D:\\NYU_Files\\2025 SPRING\\Summer_Research\\新\\PYTHON\\QWEN\\dummy_files"
    else:
        print("running on hpc")
        BASE_DIR = "/gpfsnyu/scratch/zg2598/Qwen/OUT/COMMUNICATION_LOG/" # base directory for the pt files
    
    
    Logger = ErrorLogger()
    avail_pt = scan_pt(base_dir=BASE_DIR)
    print(f"{len(avail_pt)} to process...")

    # expected return is :{"name": name,"shape": shape, "standard": std, "mean": mu}
    # so the field names will be:
    fieldnames = ["name", "gamma", "beta", "mu"]

    chunk_size = int(5e2) # 分块，防止内存崩溃
    
    
    # ✅ 修复缩进问题
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in tqdm(range(0, len(avail_pt), chunk_size), desc="Batch Processing"):
            batch = avail_pt[i:i+chunk_size]
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(cal_distribution, 
                                           path, 
                                           device=device,
                                           sample_enabled=sample_enabled,
                                           sample_size=sample_size,
                                           to64=to64) 
                           for path in batch]

                for future in tqdm(as_completed(futures), total=len(futures)):
                    try:
                        ans = future.result()
                        writer.writerow(ans)
                        csvfile.flush()
        
                    except Exception as e:
                        print(f"[Error] One task failed: {e}")
                        Logger.record(e)

    return csv_path
if __name__ == "__main__":
    csv_path = main()
    print(f"ALL FILE PRPROCESSED, CHECK\n{csv_path}\nTO SEE THE RESULT")