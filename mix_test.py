import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.special import expit
from numba import jit,vectorize,int64,float64,carray

load_input = np.load("nn_data/mix_input.npy")[0:1500]
load_class = np.load("nn_data/mix_class.npy")[0:1500]

# Input ve Output verilerini karıştırdık
s_mix_data = np.arange(load_input.shape[0])
np.random.shuffle(s_mix_data)

print("Eğitim Veri Sayısı : " +str(len(load_input)))
load_input_weight = np.load("mix_weight/input_weight.npy")
load_h1_weight = np.load("mix_weight/h1_weight.npy")
load_h1_bias = np.load("mix_weight/h1_bias.npy")
load_output_bias = np.load("mix_weight/output_bias.npy")
epoch = 1
e_tolerance = 0.05
class_count = 2

class_count_arr = np.zeros(class_count)
for i in range(len(load_class)):
    if np.array_equal(load_class[i], [0]):
        class_count_arr[0] += 1
    elif np.array_equal(load_class[i], [1]):
        class_count_arr[1] += 1

print("1. Sınıf Veri Sayısı : "+str(class_count_arr[0])+"  2. Sınıf Veri Sayısı : "+str(class_count_arr[1]))
print("  Hata Toleransı : "+str(e_tolerance)+"  Epoch : "+str(epoch))

input_count = len(load_input[0])
hidden_count = 40
out_count = 1

#Sigmoid aktivasyon fonksiyonu
def act_sig(x):
    act = 1/(1+np.exp(-1*x))
    return act_step(act)

def act_step(x):
    if x < 0.2:
        return 0
    elif x > 0.8:
        return 1
    else:
        return x

# RMSE hata fonksionu
def er_rmse(target,output):
    error = 0
    count = len(target)
    for i in range(count):
        error = error + (float(target[i])-output[i])**2
    return math.sqrt((1/count)*error)

# MSE hata fonksionu
def er_mse(target,output):
    error = 0
    count = len(target)
    for i in range(count):
        error = error + (target[i]-output[i])**2
    return (1/count)*error

# EĞİTİM AĞI

#Ağın başlangıç değerlerinin atanması

input_weight_delta = np.zeros((input_count,hidden_count))
h1_weight_delta = np.zeros((hidden_count,hidden_count))


#Ağırlıkları dışardan aldık
h1_bias = load_h1_bias
output_bias = load_output_bias
input_weight = load_input_weight
h1_weight = load_h1_weight

h1_bias_delta = np.zeros(hidden_count)
output_bias_delta = np.zeros(out_count)


#İLERİ BESLEME

def feedforward(input,weight,bias):
    output = list()
    w_array = np.array(weight)
    for i in range(len(bias)):
        out = 0
        for y in range(len(input)):
            out = out + (float(input[y])*w_array[y,i])

        out = act_sig(out+bias[i])
        output.append(out)
    return output

#Eğitim Sinir Ağı

def nn(e_count):
    output_error = list()
    acc_array = list()
    true_count = 0
    pre_count = 0

    for j in range(e_count):
        # Her epochda farklı sırada input ve output veriyoruz
        input_data = load_input[s_mix_data]
        target_data = load_class[s_mix_data]

        e_true_count = 0
        e_pre_count = 0
        epoch_error = list()
        print("Epoch" + str(j))
        for i in range(len(input_data)):
            h1_out = feedforward(input_data[i],input_weight,h1_bias[0])
            n_out = feedforward(h1_out,h1_weight,output_bias[0])
            n_err = er_rmse(target_data[i],n_out)
            epoch_error.append(n_err)
            # GERİ BESLEME
            if(n_err<e_tolerance):
                e_true_count += 1
            e_pre_count += 1


        # Accuracy Hesaplama
        true_count += e_true_count
        pre_count += e_pre_count
        acc_rate = int((true_count/pre_count)*100)
        ep_rate = int((e_true_count/e_pre_count)*100)
        print("Epoch True : "+str(e_true_count)+"   Epoch Rate : "+ str(ep_rate)+"%  Accuracy :  " + str(acc_rate)+"%")
        output_error.append(epoch_error)
        acc_array.append(acc_rate)

        # Doğruluk oranı kaydı ve grafik olarak gösterme
        # acc_save_array = np.array(acc_array[-1])
        # np.save("acc_test/hidden_"+str(hidden_count),acc_save_array)
        # plt.plot(acc_array)
        # plt.ylabel("Accuracy")
        # plt.show()
        # print(output_error)

if __name__ == "__main__":
    nn(epoch)

