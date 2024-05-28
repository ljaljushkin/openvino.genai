import re
from pathlib import Path
import pandas as pd

LOG_FILE = '/home/nlyaly/projects/openvino.genai/llm_bench/python/phi-3b-lora-perf.log'
data = []
for line in open(LOG_FILE):
    if '[ INFO ] Model path=/' in line:
        exp_name = Path(line.split('Model path=')[1].split(',')[0]).name
        exp_name = exp_name.replace('int4_sym_g128_r100_data_', '')
        exp_name = exp_name.replace('lora_int8', 'lora100_int8')
        if exp_name == 'lora':
            exp_name = 'lora100'
        print(exp_name)

    if '[ INFO ] [Average]' in line:
        all_matches = re.findall(r'(\d+\.\d+)\s+(ms/token|tokens/s)', line)
        print("All numbers:", all_matches)
        data.append({
            'Name': exp_name,
            'FTL': float(all_matches[0][0]),
            'STL': float(all_matches[1][0]),
            'STT': float(all_matches[2][0]),
        })

df = pd.DataFrame(data)
# print(df)
dyn_df = df[df['FTL'] < 100]
mean_df = dyn_df.groupby('Name').agg(
    mean_FTL=('FTL', 'mean'),
    mean_STL=('STL', 'mean'),
    mean_STT=('STT', 'mean')
)
mean_df = mean_df.sort_values(by=['mean_STT'], ascending=False)
mean_df = mean_df[mean_df['Name'] in ['lora100', 'lora100_int8']]
pd.options.display.float_format = '{:,.2f}'.format
print(mean_df)