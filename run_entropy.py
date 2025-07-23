# run
import os
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from tqdm import tqdm
import multiprocessing
import traceback

from utilities.cal_entropy import cal_entropy
from utilities.utilities import read_pt_from_csv,scan_pt
from utilities.ErrorLogger import ErrorLogger 




# def main():
#     base_dir = os.path.dirname(__file__)
#     output_path = os.path.join(base_dir,"data_obtained")
#     os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists
#     BASE_DIR = "/gpfsnyu/scratch/zg2598/Qwen/OUT/COMMUNICATION_LOG/"
#     Logger = ErrorLogger()

#     print("total cpu:", multiprocessing.cpu_count())
#     max_workers=max(4,multiprocessing.cpu_count()-2)
#     print(f"working on: {max_workers}")
#     if not torch.cuda.is_available():
#         print("local test")
#     else:
#         print("running on hpc")
        
#     # Cache file paths
#     CACHE_FILE = os.path.join(output_path, "zzz_avail_pt.csv")
#     RESULTS_FILE = "005_ENTROPY_RESULTS_PROCESSPOOL.csv"
#     fieldnames = ["name", "entropy"]  # Fixed typo in "entorpy"

#     avail_pt = read_pt_from_csv(BASE_DIR)
#     print(f"{len(avail_pt)} to process...")
    
#     # cache the avail_pt to os.path.join(output_path,"zzz.csv")
#     cache_file = os.path.join(output_path, "zzz_avail_pt.csv")

#     # 将文件列表缓存到 CSV（如果列表非空）
#     if avail_pt:
#         try:
#             # 建议使用 pandas 保存 CSV（更规范）
#             import pandas as pd
#             pd.DataFrame({"name": avail_pt}).to_csv(cache_file, index=False)
#             print(f"File list cached to: {cache_file}")
#         except Exception as e:
#             print(f"Failed to cache file list: {str(e)}")
#     else:
#         print("Warning: No .pt files found to cache!")
    
#     csv_path ="005_ENTROPY_RESULTS_PROCESSPOOL.csv"
#     fieldnames = ["name", "entorpy"]

#     chunk_size = int(5e2) # 分块，防止内存崩溃
    
    
#     # ✅ 修复缩进问题
#     with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         for i in tqdm(range(0, len(avail_pt), chunk_size), desc="Batch Processing"):
#             batch = avail_pt[i:i+chunk_size]
#             with ProcessPoolExecutor(max_workers=max_workers) as executor:
#                 futures = [executor.submit(cal_entropy, os.path.join(BASE_DIR,path)) for path in batch]

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
    
import os
import csv
import torch
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

def main():
    # Initialize paths and directories
    base_dir = os.path.dirname(__file__)
    output_path = os.path.join(base_dir, "data_obtained")
    os.makedirs(output_path, exist_ok=True)
    BASE_DIR = "/gpfsnyu/scratch/zg2598/Qwen/OUT/COMMUNICATION_LOG/"
    Logger = ErrorLogger()

    # System configuration
    total_cpus = multiprocessing.cpu_count()
    print(f"Total CPU cores available: {total_cpus}")
    max_workers = max(4, total_cpus - 2)
    print(f"Using {max_workers} worker processes")
    print("Running in local test mode" if not torch.cuda.is_available() else "Running on HPC")

    # File paths configuration
    CACHE_FILE = os.path.join(output_path, "zzz_avail_pt.csv")
    RESULTS_FILE = "001_ENTROPY_RESULTS_PROCESSPOOL.csv"
    fieldnames = ["name", "entropy"]
    chunk_size = 250  # Process files in batches to prevent memory issues

    # Load or create results file
    processed_files = set()
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r') as f:
                reader = csv.DictReader(f)
                processed_files = {row['name'] for row in reader}
            print(f"Found {len(processed_files)} previously processed files")
        except Exception as e:
            print(f"Error reading results file: {e}")
            Logger.record(e)

    # Get all available files and filter out processed ones
    try:
        all_files = scan_pt(BASE_DIR)
        avail_pt = [f for f in all_files if f not in processed_files]
        print(f"Total files found: {len(all_files)}, Remaining to process: {len(avail_pt)}")
        
        if not avail_pt:
            print("All files already processed!")
            return RESULTS_FILE

        # Cache the file list
        pd.DataFrame({"name": avail_pt}).to_csv(CACHE_FILE, index=False)
        print(f"Cached file list to: {CACHE_FILE}")

    except Exception as e:
        print(f"Error loading files: {e}")
        Logger.record(e)
        return None

    # Process files in batches
    try:
        with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not processed_files:  # Write header only for new file
                writer.writeheader()

            for i in tqdm(range(0, len(avail_pt), chunk_size), desc="Processing batches"):
                batch = avail_pt[i:i + chunk_size]
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(cal_entropy, os.path.join(BASE_DIR, path),
                                        fp64_enable =  True, 
                                        pure_data_enable = True, 
                                        scaling = int(1e6))
                        for path in batch
                    }

                    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                        path = futures[future]
                        try:
                            result = future.result()
                            writer.writerow(result)
                            csvfile.flush()  # Ensure data is written after each file
                        except Exception as e:
                            print(f"Error processing {path}: {e}")
                            Logger.record(f"Failed {path}: {str(e)}")

    except Exception as e:
        print(f"Fatal processing error: {e}")
        Logger.record(f"Fatal error: {str(e)}")
        return None

    print(f"\nProcessing complete! Results saved to {RESULTS_FILE}")
    return RESULTS_FILE

if __name__ == "__main__":
    result_file = main()
    if result_file:
        print(f"ALL FILES PROCESSED, CHECK\n{result_file}\nFOR RESULTS")
    else:
        print("Processing failed - check logs for errors")



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