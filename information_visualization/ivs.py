# ivs.py

import matplotlib.pyplot as plt
import pandas as pd

data = "007_ENTROPY_RATIO.csv"
df = pd.read_csv(data)
x = df['entorpy'] # 笔误
required_fields = ['ratio_theory', 'ratio_os']
for field in required_fields:
    if field not in df.columns:
        raise ValueError(f"Missing required field: {field}")
    
    y = df[field]
    plt.scatter(x, y)
    plt.xlabel('entropy')
    plt.ylabel(field)
    plt.title(f'entropy vs. {field}')
    plt.grid(True)
    plt.savefig(f'{field}_vs_entropy.png')
    plt.clf()

