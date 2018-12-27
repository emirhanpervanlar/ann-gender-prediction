import numpy as np
import random
i_array = np.load("nn_data/nn_input.npy")
c_array = np.load("nn_data/class_array.npy")
f_count = len(i_array)
c_count = len(c_array)
# t_count = f_count + c_count




s_mix_data = np.arange(i_array.shape[0])
np.random.shuffle(s_mix_data)

input_data = i_array[s_mix_data]
target_data = c_array[s_mix_data]

np.save("nn_data/mix_input", input_data)
np.save("nn_data/mix_class", target_data)