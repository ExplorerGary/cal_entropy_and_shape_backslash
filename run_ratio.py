# run
import os
import torch
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from tqdm import tqdm
import multiprocessing
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    from utilities.cal_ratio import cal_ratio
except:
    from .utilities.cal_ratio import cal_ratio
    
from utilities.utilities import read_pt_from_csv,scan_pt
from utilities.ErrorLogger import ErrorLogger 


suffix_dict = {int(1e6):"b",
               int(1e5):"c",
               int(1e4):"d",}

def main(scale = int(1e6)):
    info = f'''scale -- {scale}'''
    # 初始化路径
    base_dir = os.path.dirname(__file__)
    output_path = os.path.join(base_dir, "data_obtained")
    os.makedirs(output_path, exist_ok=True)
    BASE_DIR = "/gpfsnyu/scratch/zg2598/Qwen/OUT/COMMUNICATION_LOG/" if torch.cuda.is_available() else "D:\\NYU_Files\\2025 SPRING\\Summer_Research\\新\\PYTHON\\QWEN\\dummy_files" 
    Logger = ErrorLogger()

    # 系统信息
    total_cpus = multiprocessing.cpu_count()
    print(f"Total CPU cores available: {total_cpus}")
    max_workers = max(4, total_cpus - 2)
    print(f"Using {max_workers} worker processes")
    print("Running in local test mode" if not torch.cuda.is_available() else "Running on HPC")

    # 文件配置
    try:
        suffix = suffix_dict[int(scale)]
    except:
        suffix  = f"e_scaling_{scale}"
        
    cache_file = os.path.join(output_path, f"zzz_avail_pt_{suffix}.csv")
    results_file = os.path.join(output_path,f"001_ENTROPY_RESULTS_PROCESSPOOL_{suffix}.csv")
    
    fieldnames = ["name", 
                  "compressed_bit_size",
                  "original_bit_theory",
                  "avg_bit_per_entry",
                  "time_used",]

    
    
    chunk_size = 250

    processed_files = set()
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                reader = csv.DictReader(f)
                processed_files = {row['name'] for row in reader}
            print(f"Found {len(processed_files)} previously processed files for mode {suffix}")
        except Exception as e:
            print(f"Error reading results file: {e}")
            Logger.record(e)

    try:
        all_files = scan_pt(BASE_DIR)
        avail_pt = [f for f in all_files if f not in processed_files]
        print(f"Total files found: {len(all_files)}, Remaining to process: {len(avail_pt)}")

        if not avail_pt:
            print("All files already processed!")
            return results_file

        pd.DataFrame({"name": avail_pt}).to_csv(cache_file, index=False)
        print(f"Cached file list to: {cache_file}")

    except Exception as e:
        print(f"Error loading files: {e}")
        Logger.record(e)
        return None

    try:
        with open(results_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not processed_files:
                writer.writeheader()

            for i in tqdm(range(0, len(avail_pt), chunk_size), desc=f"Processing batches ({suffix})"):
                batch = avail_pt[i:i + chunk_size]

                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            cal_ratio,
                            os.path.join(BASE_DIR, path),
                            scaling=scale,
                        ): path
                        for path in batch
                    }

                    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                        path = futures[future]
                        try:
                            result = future.result()
                            writer.writerow(result)
                            csvfile.flush()
                        except Exception as e:
                            print(f"Error processing {path}: {e}")
                            Logger.record(f"Failed {path}: {str(e)}")

    except Exception as e:
        print(f"Fatal processing error: {e}")
        Logger.record(f"Fatal error: {str(e)}")
        return None

    print(f"\nProcessing complete! Results saved to {results_file}")
    return results_file


if __name__ == "__main__":
    result_file_b = main(scale = int(1e6))
    if result_file_b:
        print(f"\n[001b] PURE_DATA ENABLED:\nCheck results in {result_file_b}")
    else:
        print("[001]Failed — check logs.")

    result_file_c = main(scale = int(1e5))
    if result_file_c:
        print(f"\n[001c] PURE_DATA ENABLED:\nCheck results in {result_file_c}")
    else:
        print("[001]Failed — check logs.")
    
    result_file_d = main(scale = int(1e4))
    if result_file_d:
        print(f"\n[001c] PURE_DATA ENABLED:\nCheck results in {result_file_d}")
    else:
        print("[001]Failed — check logs.")




