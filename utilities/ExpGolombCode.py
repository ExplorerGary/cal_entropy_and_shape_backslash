import torch
import numpy as np

class ExpGolombCode:
    """
    这是由吴俊师兄实现的整数序列的指数-Golomb编码（Exp-Golomb Code）。

    参数:
        k (int): Exp-Golomb编码的阶数，默认为0。
    方法:
        encode(nums):
            将整数列表编码为对应的Exp-Golomb码字（字符串形式）。
        encode_fast(nums):
            使用PyTorch高效计算整数张量的前导零数量和编码值。
        decode(codes):
            将Exp-Golomb码字（字符串列表）解码还原为原始整数。
        streamEncode(nums):
            将整数列表编码为一个拼接的比特串。
        streamDecode(streamStr):
            将拼接的比特串解码还原为原始整数列表。
    示例:
        >>> egc = ExpGolombCode(k=0)
        >>> codes = egc.encode([3, 7])
        >>> nums = egc.decode(codes)
    
    注意：
        不支持负数
    """
    def __init__(self, k=0):
        self.k = k

    def encode(self, nums):
        codes = [None] * len(nums)
        for i, num in enumerate(nums):
            code = num + (1 << self.k)
            codes[i] = '0' * (int(code).bit_length() - self.k - 1) + bin(code)[2:]
        return codes

    def encode_fast(self, nums):
        codes = nums + (1 << self.k)
        zeros = torch.ceil(torch.log2(codes)) - self.k - 1
        return zeros, codes

    def decode(self, codes):
        nums = torch.zeros(len(codes), dtype=torch.long)
        for i, code in enumerate(codes):
            num = int("0b" + code, base=2)
            nums[i] = num - (1 << self.k)
        return nums

    def streamEncode(self, nums):
        codes = self.encode(nums)
        return "".join(codes)

    def streamDecode(self, streamStr):
        codes = []
        start = 0
        while start < len(streamStr):
            cnt = 0
            while streamStr[start + cnt] == "0":
                cnt += 1
            end = start + 2 * cnt + self.k + 1
            codes.append(streamStr[start:end])
            start = end
        nums = self.decode(codes)
        return nums

def local_test():
    """
    ExpGolombCode类的本地测试。
    演示整数序列的编码与解码过程。
    """
    egc = ExpGolombCode(k=0)
    nums = [0, 1, 2, 3, 4, 5, 10, 20]
    # 使用 torch.Tensor 测试
    nums_torch = torch.tensor(nums)
    print("使用 torch.Tensor 测试:")
    codes_torch = egc.encode(nums_torch.tolist())
    print("编码结果:", codes_torch)
    decoded_torch = egc.decode(codes_torch)
    print("解码结果:", decoded_torch.tolist())

    # 使用 np.ndarray 测试
    nums_np = np.array(nums)
    print("使用 np.ndarray 测试:")
    codes_np = egc.encode(nums_np.tolist())
    print("编码结果:", codes_np)
    decoded_np = egc.decode(codes_np)
    print("解码结果:", decoded_np.tolist())
    print("原始数字:", nums)
    codes = egc.encode(nums)
    print("编码结果:", codes)
    decoded = egc.decode(codes)
    print("解码结果:", decoded.tolist())
    stream = egc.streamEncode(nums)
    print("流式编码:", stream)
    stream_decoded = egc.streamDecode(stream)
    print("流式解码:", stream_decoded.tolist())

# local_test()