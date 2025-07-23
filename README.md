# 重构的 Compression Check工作区：

**文件**
000a,b,c.txt：debug文件
001.csv: entropy
002.csv: shape parameter
003/csv: 原始bucket的理论和实际大小(单位为bit)
004/csv：经过EG压缩后的bucket的大小（单位为bit）和平均码长（单位为biit）
005.csv: bucket里面数据的最大几个值（90,95,100百分位数） -- 已取abs
zzz.csv: 经过扫盘得出的.pt文件列表，以csv格式存储，到时候会merge一下，方便找出失败的格式







