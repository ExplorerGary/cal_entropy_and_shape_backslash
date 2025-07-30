# EG encoding: 
try:
    from ExpGolombCode import ExpGolombCode as EGCode
    from make_EGcodeTable import make_EGTable_with_k
except:
    from .ExpGolombCode import ExpGolombCode as EGCode
    from .make_EGcodeTable import make_EGTable_with_k
import os
import torch
import numpy as np
from tqdm import tqdm
from bitarray import bitarray
import pickle


def load_EG_table_k(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)  # 加载非常快


class ExpGolombEncoding(EGCode):
    def __init__(self, 
                 k:int = 0,
                 range:int = 200000):
        super().__init__(k)
        
        expected_length = 2 * range + 1
        # loading and evalingating the EG_table_k.pkl
        base_dir = os.path.dirname(__file__)
        save_dir = os.path.join(base_dir,"..","data_to_use","EG_code_table")
        file_path = os.path.join(save_dir,f"EG_table_{k}.pkl")
        try:
            EG_table = load_EG_table_k(file_path)
            if len(EG_table) != expected_length:
                print("possible unmatch detected, regenerating...")
                save_file_path = make_EGTable_with_k(k = k)
            self.code_table = load_EG_table_k(file_path=save_file_path)
        except FileNotFoundError:
            print("File don't exist, generating...")
            save_file_path = make_EGTable_with_k(k = k)
            self.code_table = load_EG_table_k(file_path=save_file_path)
        
        print(f"EG_table_{k} is loaded, using the highspeed: encode_table...")
            
    def encode_table(self,nums):
        codes = None
        return codes
    
    
    
    def encode_retro(self, nums):
        codes = [None] * len(nums)
        for i, num in enumerate(nums):
            # 加入signbit逻辑：
            sign_bit = "0" if num>=0 else "1"
            num = abs(num)
            code = num + (1 << self.k)
            codes[i] = '0' * (int(code).bit_length() - self.k - 1) + bin(code)[2:] + sign_bit
        return codes
    def encode(self, nums, debug:bool = False):  # 改进为tolist实现，但是还是有进一步提升的空间
        
        codes = [None] * len(nums)
        nums = nums.tolist() if isinstance(nums, torch.Tensor) else nums
        iterator = tqdm(nums, desc="Processing") if debug else nums
        for i, num in enumerate(iterator):
            sign_bit = "0" if num >=0 else "1"
            num = abs(num)
            code = num + (1 << self.k)
            codes[i] = '0' * (int(code).bit_length() - self.k - 1) + bin(code)[2:] + sign_bit            
         
        return codes
    
    def encode_bitarray(self, nums: torch.Tensor) -> bitarray:
        if not torch.is_tensor(nums):
            nums = torch.tensor(nums)

        signs = (nums >= 0).int()
        abs_vals = torch.abs(nums)
        codes = bitarray(endian='big')

        for val, sign in zip(abs_vals.tolist(), signs.tolist()):
            code = val + (1 << self.k)
            prefix_len = int(code).bit_length() - self.k - 1
            prefix = '0' * prefix_len
            body = bin(code)[2:]
            codes.extend(prefix + body + str(1 - sign))  # 正→0，负→1，和你原版一致

        return codes

        
    
    def streamEncode(self, nums):
        return super().streamEncode(nums=nums)
    
    # =============================================
    
    def decode(self, codes):
        nums = torch.zeros(len(codes), dtype=torch.long)
        for i, code in enumerate(codes):
            sign = 1 if code[-1] == "0" else -1
            num = int("0b" + code[:-1], base=2) 
            nums[i] = sign * (num - (1 << self.k))
        return nums

    def streamDecode(self, streamStr):
        codes = []
        start = 0
        while start < len(streamStr):
            cnt = 0
            while streamStr[start + cnt] == "0":
                cnt += 1
            end = start + 2 * cnt + self.k + 1 + 1 # 再加1是因为signbit
            codes.append(streamStr[start:end])
            start = end
        nums = self.decode(codes)
        return nums

def test():
    nums = [0,1,-1]
    for _ in range(2,4):
        nums.extend([_,-1*_])
    
    nums = torch.tensor(nums)
    
    print("nums:")
    print(nums)

    # streamencoding and stream decoding test:
    EG = ExpGolombEncoding(k=0)
    encoded_str = EG.streamEncode(nums)
    decoded_str = EG.streamDecode(encoded_str)
    
    print(f"{'They are equal' if np.array_equal(nums,decoded_str) else 'They are not equal'}")
    
    
    
    
    
    # EG = ExpGolombEncoding(k=0)
    # encoded = EG.encode(nums)
    # print(f"encoded:\n{encoded}")
    # decoded = EG.decode(encoded)
    # print(f"decoded:\n{decoded}")
    # print(f"{'They are equal' if np.array_equal(nums,decoded) else 'They are not equal'}")
    # print("+++++++++++++++++++++++++++++")
    # nums = [0,1,-1]
    
    # i = 100
    # import random
    # for _ in range(10):
    #     num = random.randint(0, i)
    #     nums.extend([num, -1 * num]) 

    
    # nums = torch.tensor(nums)
    # encoded = EG.encode(nums)
    # print(f"encoded:\n{encoded}")
    # decoded = EG.decode(encoded)
    # print(f"decoded:\n{decoded}")
    # print(f"{'They are equal' if np.array_equal(nums,decoded) else 'They are not equal'}")




# test()    
######################################################################################################################################
def local_test():
    from bitarray.util import ba2int
    from tqdm import tqdm

    # === 构造数据 ===
    def generate_random_tensor(length=60_000_000, max_val=104000):
        return torch.randint(low=0, high=max_val + 1, size=(length,), dtype=torch.int32)

    # 用法示例：
    test_data = generate_random_tensor()
    print(f"✅ Total elements: {len(test_data)}")

    # === 初始化编码器 ===
    EG = ExpGolombEncoding(k=0)

    # # === 原始字符串编码 ===
    # print("\n📦 Encoding with string version...")
    # codes_str = []
    # for num in tqdm(test_data.tolist(), desc="String Encoding"):
    #     sign_bit = "0" if num >= 0 else "1"
    #     val = abs(num)
    #     code = val + (1 << EG.k)
    #     prefix = '0' * (code.bit_length() - EG.k - 1)
    #     body = bin(code)[2:]
    #     codes_str.append(prefix + body + sign_bit)
    # concat_str = ''.join(codes_str)

    # === bitarray 编码 ===
    print("\n📦 Encoding with bitarray version...")
    import time
    start = time.time()
    codes_bitarray = EG.encode_bitarray(test_data)
    end = time.time()
    concat_bitstr = codes_bitarray.to01()
    print(f"📦 Bitarray version takes {end-start} seconds")
    # # === 比较结果 ===
    # print("\n🔍 Comparing outputs...")
    # if concat_str == concat_bitstr:
    #     print(f"✅ 完全一致！总长度：{len(concat_str)} bits")
    # else:
    #     print(f"❌ 不一致！长度差异：str={len(concat_str)} vs bitarray={len(concat_bitstr)}")
    #     # 找到第一个不同的位置
    #     for i, (s, b) in enumerate(zip(concat_str, concat_bitstr)):
    #         if s != b:
    #             print(f"Mismatch at bit {i}: str={s} vs bitarray={b}")
    #             break



def split_bitarray_str(bit_str, ref_codes):
    """辅助函数：根据参考编码长度，对bitarray拼接字符串切片"""
    res = []
    idx = 0
    for code in ref_codes:
        res.append(bit_str[idx:idx + len(code)])
        idx += len(code)
    return res


# local_test()

# # EG encoding: 
# try:
#     from ExpGolombCode import ExpGolombCode as EGCode
# except:
#     from .ExpGolombCode import ExpGolombCode as EGCode
# import os
# import torch
# import numpy as np
# from tqdm import tqdm
# from bitarray import bitarray

# class ExpGolombEncoding(EGCode):
#     def __init__(self, k=0):
#         super().__init__(k)
    
#     def encode(self, nums, debug:bool = False):  # 改进为tolist实现，但是还是有进一步提升的空间
        
#         codes = [None] * len(nums)
#         nums = nums.tolist() if isinstance(nums, torch.Tensor) else nums
#         iterator = tqdm(nums, desc="Processing") if debug else nums
#         for i, num in enumerate(iterator):
#             sign_bit = "0" if num >=0 else "1"
#             num = abs(num)
#             code = num + (1 << self.k)
#             codes[i] = '0' * (int(code).bit_length() - self.k - 1) + bin(code)[2:] + sign_bit            
         
#         return codes
    
#     def encode_bitarray(self, nums: torch.Tensor) -> bitarray:
#         if not torch.is_tensor(nums):
#             nums = torch.tensor(nums)

#         signs = (nums >= 0).int()
#         abs_vals = torch.abs(nums)
#         codes = bitarray(endian='big')

#         for val, sign in zip(abs_vals.tolist(), signs.tolist()):
#             code = val + (1 << self.k)
#             prefix_len = int(code).bit_length() - self.k - 1
#             prefix = '0' * prefix_len
#             body = bin(code)[2:]
#             codes.extend(prefix + body + str(1 - sign))  # 正→0，负→1，和你原版一致

#         return codes

        
    
#     def streamEncode(self, nums):
#         return super().streamEncode(nums=nums)
    
#     # =============================================
    
#     def decode(self, codes):
#         nums = torch.zeros(len(codes), dtype=torch.long)
#         for i, code in enumerate(codes):
#             sign = 1 if code[-1] == "0" else -1
#             num = int("0b" + code[:-1], base=2) 
#             nums[i] = sign * (num - (1 << self.k))
#         return nums

#     def streamDecode(self, streamStr):
#         codes = []
#         start = 0
#         while start < len(streamStr):
#             cnt = 0
#             while streamStr[start + cnt] == "0":
#                 cnt += 1
#             end = start + 2 * cnt + self.k + 1 + 1 # 再加1是因为signbit
#             codes.append(streamStr[start:end])
#             start = end
#         nums = self.decode(codes)
#         return nums

# def test():
#     nums = [0,1,-1]
#     for _ in range(2,4):
#         nums.extend([_,-1*_])
    
#     nums = torch.tensor(nums)
    
#     print("nums:")
#     print(nums)

#     # streamencoding and stream decoding test:
#     EG = ExpGolombEncoding(k=0)
#     encoded_str = EG.streamEncode(nums)
#     decoded_str = EG.streamDecode(encoded_str)
    
#     print(f"{'They are equal' if np.array_equal(nums,decoded_str) else 'They are not equal'}")
    
    
    
    
    
#     # EG = ExpGolombEncoding(k=0)
#     # encoded = EG.encode(nums)
#     # print(f"encoded:\n{encoded}")
#     # decoded = EG.decode(encoded)
#     # print(f"decoded:\n{decoded}")
#     # print(f"{'They are equal' if np.array_equal(nums,decoded) else 'They are not equal'}")
#     # print("+++++++++++++++++++++++++++++")
#     # nums = [0,1,-1]
    
#     # i = 100
#     # import random
#     # for _ in range(10):
#     #     num = random.randint(0, i)
#     #     nums.extend([num, -1 * num]) 

    
#     # nums = torch.tensor(nums)
#     # encoded = EG.encode(nums)
#     # print(f"encoded:\n{encoded}")
#     # decoded = EG.decode(encoded)
#     # print(f"decoded:\n{decoded}")
#     # print(f"{'They are equal' if np.array_equal(nums,decoded) else 'They are not equal'}")




# # test()    

# def local_test():
#     from bitarray.util import ba2int
#     from tqdm import tqdm

#     # === 构造数据 ===
#     def generate_random_tensor(length=60_000_000, max_val=104000):
#         return torch.randint(low=0, high=max_val + 1, size=(length,), dtype=torch.int32)

#     # 用法示例：
#     test_data = generate_random_tensor()
#     print(f"✅ Total elements: {len(test_data)}")

#     # === 初始化编码器 ===
#     EG = ExpGolombEncoding(k=0)

#     # # === 原始字符串编码 ===
#     # print("\n📦 Encoding with string version...")
#     # codes_str = []
#     # for num in tqdm(test_data.tolist(), desc="String Encoding"):
#     #     sign_bit = "0" if num >= 0 else "1"
#     #     val = abs(num)
#     #     code = val + (1 << EG.k)
#     #     prefix = '0' * (code.bit_length() - EG.k - 1)
#     #     body = bin(code)[2:]
#     #     codes_str.append(prefix + body + sign_bit)
#     # concat_str = ''.join(codes_str)

#     # === bitarray 编码 ===
#     print("\n📦 Encoding with bitarray version...")
#     import time
#     start = time.time()
#     codes_bitarray = EG.encode_bitarray(test_data)
#     end = time.time()
#     concat_bitstr = codes_bitarray.to01()
#     print(f"📦 Bitarray version takes {end-start} seconds")
#     # # === 比较结果 ===
#     # print("\n🔍 Comparing outputs...")
#     # if concat_str == concat_bitstr:
#     #     print(f"✅ 完全一致！总长度：{len(concat_str)} bits")
#     # else:
#     #     print(f"❌ 不一致！长度差异：str={len(concat_str)} vs bitarray={len(concat_bitstr)}")
#     #     # 找到第一个不同的位置
#     #     for i, (s, b) in enumerate(zip(concat_str, concat_bitstr)):
#     #         if s != b:
#     #             print(f"Mismatch at bit {i}: str={s} vs bitarray={b}")
#     #             break



# def split_bitarray_str(bit_str, ref_codes):
#     """辅助函数：根据参考编码长度，对bitarray拼接字符串切片"""
#     res = []
#     idx = 0
#     for code in ref_codes:
#         res.append(bit_str[idx:idx + len(code)])
#         idx += len(code)
#     return res


# # local_test()