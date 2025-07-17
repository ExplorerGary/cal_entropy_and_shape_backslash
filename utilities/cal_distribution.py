import torch
import numpy as np
import os
from .utilities import read_pt, to_int

rdo = 2e3
clip = 1
# 当前这个文件的路径
base_dir = os.path.dirname(__file__)

# 拼接 gamma_table.pt 的绝对路径
gamma_table_path = os.path.abspath(os.path.join(base_dir, "..", "data_to_use", "gamma_table.pt"))
rgamma_table_path = os.path.abspath(os.path.join(base_dir, "..", "data_to_use", "r_gamma_table.pt"))

# 加载
gamma_table = torch.load(gamma_table_path)
r_gamma_table = torch.load(rgamma_table_path)



def cal_distribution(pt_path:str,device=None,sample_enabled:bool=False,sample_size:int = 10000, to64 = True) -> dict:
    """
    params:
    pt_path: str, the path to the pt file
    device: torch.device, the device to use, default is None
    sample_enabled: bool, whether to enable sampling, default is False
    sample_size: int, the size of the sample to use, default is 10000
    to64: bool, whether to convert the tensor to float64, default is True
    
    
    return: dict, the output is a dict
    {
        "shape": shape,
        "standard": standard,
        "mean": mean
    }
    """
    name = os.path.basename(pt_path)
    with torch.no_grad():
        arr = read_pt(pt_path) # pt_array is a dict
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cuda:0")
        # Evaluate the shape parameter        
        # the input is a single dimensional numpy array
        
        arr = torch.tensor(arr).to(device).to(torch.float64)
        if sample_enabled: # ramdoming pick sample_size elements
            if sample_size <= arr.shape[0]:
                # random pick sample_size elements
                torch.manual_seed(42)  # for reproducibility
                indices = torch.randperm(arr.shape[0])[:sample_size]
                arr = arr[indices]
                
        n = arr.shape[0]
        var = torch.sum((arr ** 2).to(device))
        mean = torch.sum(torch.abs(arr).to(device))
        
        r_gamma = (n * var / mean ** 2).to(device=torch.device("cpu"))
        
        # find the closest value in r_gamma_table
        pos = torch.argmin(torch.abs(r_gamma - r_gamma_table))
        shape = gamma_table[pos]
        std = torch.sqrt(var / n)
        n = torch.tensor(n)

        mu = torch.mean(arr).to(device)
        
        distribution = {"name": name,"gamma": shape, "beta": std, "mu": mu}
        
    return distribution

def cal_distribution2(pt_path:str,device=None,sample_enabled:bool=False,sample_size:int = 10000,to64:bool = True) -> dict:
    """
    return: dict, the output is a dict
    {
        "shape": shape,
        "standard": standard,
        "mean": mean
    }
    """
    name = os.path.basename(pt_path)
    with torch.no_grad():
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pt_dict = torch.load(pt_path, map_location=device)  # pt_array is a dict
        # # print(pt_dict)
        # for i in range(len(list(pt_dict.values()))):
        #     print(list(pt_dict.values())[i].shape)
            
        # Evaluate the shape parameter
        n, var, mean = 0, 0, 0   
             
        for grad in pt_dict.values():
            if to64:
                grad = grad.flatten().detach().to(torch.float64)
            else:
                grad = grad.flatten().detach()
            n += grad.shape[0]
            var += torch.sum((grad ** 2).to(device))
            mean += torch.sum(torch.abs(grad).to(device))
            
        r_gamma = (n * var / mean ** 2).to(device)
        pos = torch.argmin(torch.abs(r_gamma - r_gamma_table))
        shape = gamma_table[pos]
        std = torch.sqrt(var / n)
        n = torch.tensor(n)
        mu = torch.mean(torch.cat([v.flatten() for v in pt_dict.values()])).to(device)
        distribution = {"name": name,"gamma": shape, "beta": std, "mu": mu}

    return distribution
        # # Evaluate the shape parameter
        # n, var, mean = 0, 0, 0
        # for grad in pt_dict.values():
        #     grad = grad.flatten().detach()
        #     n += grad.shape[0]
        #     var += torch.sum((grad ** 2).to(device))
        #     mean += torch.sum(torch.abs(grad).to(device))
        # r_gamma = (n * var / mean ** 2).to(device=torch.device("cpu"))
        # pos = torch.argmin(torch.abs(r_gamma - r_gamma_table))
        # shape = gamma_table[pos]
        # std = torch.sqrt(var / n)
        # n = torch.tensor(n)
        # distribution = {"shape": shape, "standard": std, "mean": mean}




def pltter(pt_path, distribution):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy.stats import gennorm
    import os

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    arr = read_pt(pt_path)
    sns.histplot(arr, bins=256, kde=True, stat="density", label="Histogram")

    beta = float(distribution.get('standard', 0))
    shape = float(distribution.get('shape', 0))
    mean  = float(distribution.get('mean', 0))

    if beta <= 0:
        print(f"[警告] beta（标准差）={beta}，设置为 1e-6")
        beta = 1e-6

    if shape <= 0:
        print(f"[警告] shape（形状参数）={shape}，设置为 1e-6")
        shape = 1e-6

    x = np.linspace(np.min(arr), np.max(arr), 1000)
    y = gennorm.pdf(x, shape, loc=mean, scale=beta)

    if np.any(np.isinf(y)) or np.any(np.isnan(y)):
        print("[警告] GGD 拟合曲线中含有 inf 或 nan，跳过绘图")
        return

    plt.plot(x, y, color='red', label='GGD Fit')
    plt.legend()
    plt.title(f"Distribution: {os.path.basename(pt_path)}")
    plt.show()

# def pltter(pt_path, distribution):
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     from scipy.stats import gennorm

#     arr = read_pt(pt_path)
#     sns.histplot(arr, bins=256, kde=True, stat="density", label="Histogram")

#     # 绘制GGD分布曲线
#     x = np.linspace(np.min(arr), np.max(arr), 1000)
#     beta = distribution['standard']
#     shape = float(distribution['shape'])
#     if shape<=0:
#         shape = 1e-6
#     mean  = float(distribution['mean'])
#     # GGD: scipy.stats.gennorm.pdf(x, shape, loc=0, scale=beta)
#     y = gennorm.pdf(x, shape, loc=mean, scale=beta)

#     plt.plot(x, y, color='red', label='GGD Fit')
#     plt.legend()
#     plt.title(f"Distribution for {pt_path}")
#     plt.show()

def local_test():
    """
    本地测试函数
    """
    # arr = np.random.randn(1000)
    # distribution = cal_distribution(arr)
    # print(distribution)
    

    '''
    目标："D:\\NYU_Files\\2025 SPRING\\Summer_Research\\新\\PYTHON\\QWEN\\dummy_files\\R_1_E_0_S_9_B_89.pt"
    '''
    pt_path = "D:\\NYU_Files\\2025 SPRING\\Summer_Research\\新\\PYTHON\\QWEN\\dummy_files\\R_1_E_0_S_9_B_89.pt"
    # arr = read_pt("D:\\NYU_Files\\2025 SPRING\\Summer_Research\\新\\PYTHON\\QWEN\\dummy_files\\R_1_E_0_S_9_B_89.pt")
    distribution = cal_distribution(pt_path,sample_enabled=True,sample_size=10000)
    print(distribution)
    pltter("D:\\NYU_Files\\2025 SPRING\\Summer_Research\\新\\PYTHON\\QWEN\\dummy_files\\R_1_E_0_S_9_B_89.pt", distribution)

    sth = input('输入任意键继续...')
    distribution = cal_distribution(pt_path)
    print(distribution)
    pltter("D:\\NYU_Files\\2025 SPRING\\Summer_Research\\新\\PYTHON\\QWEN\\dummy_files\\R_1_E_0_S_9_B_89.pt", distribution)

    pt_path = "D:\\NYU_Files\\2025 SPRING\\Summer_Research\\新\\PYTHON\\QWEN\\dummy_files\\R_1_E_0_S_9_B_89.pt"
    # arr = read_pt("D:\\NYU_Files\\2025 SPRING\\Summer_Research\\新\\PYTHON\\QWEN\\dummy_files\\R_1_E_0_S_9_B_89.pt")
    distribution = cal_distribution(pt_path,to64=True)
    print(distribution)
    distribution = cal_distribution(pt_path=pt_path,
                                    device=None,
                                    sample_enabled=True,
                                    sample_size= 10000, 
                                    to64 = True)
    print(distribution)
    pltter("D:\\NYU_Files\\2025 SPRING\\Summer_Research\\新\\PYTHON\\QWEN\\dummy_files\\R_1_E_0_S_9_B_89.pt", distribution)
    # distribution = cal_distribution(pt_path,sample_enabled=True,sample_size=10000)
    # print(distribution)


# local_test()



