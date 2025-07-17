# cal_entropy_and_shape

本模块用于计算数据的熵（entropy）和形状（shape）相关特征，适用于数据分析和特征工程场景。

## 文件结构

1. 执行函数

    run_entropy.py

    run_shape.py

2. 关键功能函数，放在了utilities文件夹中：
    
    cal_entropy.py:
        1. 仿照matlab的entropy函数，计算一个.pt文件的entropy 【默认bins = 256】
        2. 根据GGD parameters计算理论entropy
    cal_shape.py：
        1. 对一个.pt文件进行抽样，使用scipy.fit命令，计算他的shape parameter 【默认抽样 = 3000】

            由于原始数据的dtype = fp16，可能是overflow错误发生的原因。

            我们保留了to_int，但是我们也提前做了个proprocess

    ErrorLogger.py
        1. 提供一个ErrorLogger类，记录所有的错误至data_obtained/error_log.txt



