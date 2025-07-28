# EG encoding: 
try:
    from ExpGolombCode import ExpGolombCode as EGCode
except:
    from .ExpGolombCode import ExpGolombCode as EGCode
import os
import torch
import numpy as np
from tqdm import tqdm


class ExpGolombEncoding(EGCode):
    def __init__(self, k=0):
        super().__init__(k)
    
    def encode(self, nums, debug:bool = False):
        codes = [None] * len(nums)
        if debug:
            for i, num in enumerate(tqdm(nums, desc="Processing")):
                # 加入signbit逻辑：
                sign_bit = "0" if num>=0 else "1"
                num = abs(num)
                code = num + (1 << self.k)
                codes[i] = '0' * (int(code).bit_length() - self.k - 1) + bin(code)[2:] + sign_bit
        else:
            for i, num in enumerate(nums):
                # 加入signbit逻辑：
                sign_bit = "0" if num>=0 else "1"
                num = abs(num)
                code = num + (1 << self.k)
                codes[i] = '0' * (int(code).bit_length() - self.k - 1) + bin(code)[2:] + sign_bit            
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
