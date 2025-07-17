# run
import os
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from tqdm import tqdm
import multiprocessing
import traceback

from cal_entropy import cal_entropy
from utilities import read_pt_from_csv
from ErrorLogger import ErrorLogger 




def main():
    BASE_DIR = "/gpfsnyu/scratch/zg2598/Qwen/OUT/COMMUNICATION_LOG/"
    Logger = ErrorLogger()

    print("total cpu:", multiprocessing.cpu_count())
    max_workers=max(4,multiprocessing.cpu_count()-2)
    print(f"working on: {max_workers}")
    if not torch.cuda.is_available():
        print("local test")
    else:
        print("running on hpc")
        
    # base_dir = "D:\\NYU_Files\\2025 SPRING\\Summer_Research\\新\\PYTHON\\QWEN\\dummy_files" if (not torch.cuda.is_available) else "/gpfsnyu/scratch/zg2598/Qwen/OUT/COMMUNICATION_LOG/"
    pt_csv_path = "./DATA/006_failed_to_compress.csv"
    avail_pt = read_pt_from_csv(pt_csv_path)
    print(f"{len(avail_pt)} to process...")

    
    csv_path ="005_ENTROPY_RESULTS_PROCESSPOOL.csv"
    fieldnames = ["name", "entorpy"]

    chunk_size = int(5e2) # 分块，防止内存崩溃
    
    
    # ✅ 修复缩进问题
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in tqdm(range(0, len(avail_pt), chunk_size), desc="Batch Processing"):
            batch = avail_pt[i:i+chunk_size]
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(cal_entropy, os.path.join(BASE_DIR,path)) for path in batch]

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
    
    
# # run
# import os
# import torch
# import numpy as np
# from concurrent.futures import ProcessPoolExecutor, as_completed
# import csv
# from tqdm import tqdm
# import multiprocessing
# import traceback

# from cal_ratio import cal_ratio
# from utilities import scan_pt
# from ErrorLogger import ErrorLogger 



# def main():
#     Logger = ErrorLogger()

#     print("total cpu:", multiprocessing.cpu_count())
#     max_workers=max(4,multiprocessing.cpu_count()-2)
#     print(f"working on: {max_workers}")
#     if not torch.cuda.is_available():
#         print("local test")
#     else:
#         print("running on hpc")
        
#     # base_dir = "D:\\NYU_Files\\2025 SPRING\\Summer_Research\\新\\PYTHON\\QWEN\\dummy_files" if (not torch.cuda.is_available) else "/gpfsnyu/scratch/zg2598/Qwen/OUT/COMMUNICATION_LOG/"
#     base_dir = "D:\\NYU_Files\\2025 SPRING\\Summer_Research\\新\\PYTHON\\QWEN\\dummy_files" if (not torch.cuda.is_available()) else "/gpfsnyu/scratch/zg2598/Qwen/OUT/COMMUNICATION_LOG/"
#     print(f"working on: {base_dir}")
#     avail_pt = scan_pt(base_dir=base_dir)
    
#     print(f"{len(avail_pt)} files found...")

#     scale = 1e6
#     print(f"scale: {scale}")
    
#     csv_path = os.path.join(base_dir, "004_COMPRESSION_RESULTS_PROCESSPOOL.csv")
#     fieldnames = ["name", "byte_theory", "byte_os", "byte_encoded", "ratio_theory", "ratio_os"]

#     chunk_size = 1e3 # 分块，防止内存崩溃
    
    
#     # ✅ 修复缩进问题
#     with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         for i in tqdm(range(0, len(avail_pt), chunk_size), desc="Batch Processing"):
#             batch = avail_pt[i:i+chunk_size]
#             with ProcessPoolExecutor(max_workers=max_workers) as executor:
#                 k = 0 # 或者你可以根据需要设置k的值
#                 futures = [executor.submit(cal_ratio, path, k, scale) for path in batch]

#                 for future in tqdm(as_completed(futures), total=len(futures)):
#                     try:
#                         ans = future.result()
#                         writer.writerow(ans)
#                         csvfile.flush()
        
#                     except Exception as e:
#                         print(f"[Error] One task failed: {e}")
#                         Logger.record(e)

#     return csv_path
# if __name__ == "__main__":
#     csv_path = main()
#     print(f"ALL FILE PRPROCESSED, CHECK\n{csv_path}\nTO SEE THE RESULT")




# if not torch.cuda.is_available():
#     print("local test")
# else:
#     print("running on hpc")
    
# BASE_DIR = "" if torch.cuda.is_available() else ""
# avail_pt = scan_pt(BASE_DIR)
# print(f"{len(avail_pt)} files found...")