import pandas as pd
from matplotlib import pyplot as plt

l1_noise_delta_file = '/home/nlyaly/projects/openvino.genai/llm_bench/python/l1_noise_delta.csv'
df = pd.read_csv(l1_noise_delta_file)
delta = df[df.columns[1]]

orig_l1_noise_file = '/home/nlyaly/projects/openvino.genai/llm_bench/python/original_l1_noise.csv'
df = pd.read_csv(orig_l1_noise_file)
first = df[df.columns[1]]

fig, ax = plt.subplots()
ax.plot(first, label='l1_noise_before')
ax.plot(first - delta, label='l1_noise_after')
ax.axhline(y=first.quantile(q=0.5), color ='r')
ax.set_title('L1 quantization noise with criteria')
plt.legend()
plt.show()
