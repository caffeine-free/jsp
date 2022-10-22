import pandas as pd
import os

filenames = next(os.walk('instances'), (None, None, []))[2]
run = lambda f: os.popen(f'src/main.py instances/{f}').read().strip().split()

results = [
    { 
        'file': file, 
        'makespan': run(file)[0], 
        'gap': run(file)[1] 
    } for _ in range(10) for file in sorted(filenames)
]

df = pd.DataFrame(results)
df.to_csv('out/results.csv', index=False, header=True)