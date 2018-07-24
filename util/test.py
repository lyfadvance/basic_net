from gpu_nms import gpu_nms
import numpy as np
inputs=np.array([[100.0,100.0,200.0,200.0,89.0],
                [34.0,34.0,145.0,145.0,71.0],
                [67.0,54.0,34.0,67.0,90.0],
                [102.0,99.0,187.0,191.0,23.0]],dtype=np.float32)
keep=gpu_nms(inputs,0.7)
print(keep)
