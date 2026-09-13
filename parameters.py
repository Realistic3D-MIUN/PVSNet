import os
params_m = 32
params_number_input = 1
DEVICE = "cuda"
os.makedirs("./logs",exist_ok=True)
os.makedirs("./checkpoint",exist_ok=True)
os.makedirs("./output",exist_ok=True)