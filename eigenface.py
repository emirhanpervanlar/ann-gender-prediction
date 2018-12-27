import scipy.misc as sp
import numpy as np
from os import listdir, path, makedirs
import matplotlib.pyplot as plt

img_folder = "gender_data"
out_folder= path.join(img_folder, "_imgs")
eface_folder = path.join(img_folder, "_img_eigenfaces")

#Resimleri klasörden tektek alıp listeye kaydettik
img_names = listdir(img_folder)
img_count = len(img_names)

#hangi yaş grubunda kaç resim olduğunu hesapladık
group_count = np.zeros(2)
class_array = list()
img_list = []
print(len(img_names))
for img in img_names:
    group_name = img[-5:-4]
    if group_name == "a":
        group_count[0] = group_count[0] + 1
        class_array.append([0])
    else:
        group_count[1] = group_count[1] + 1
        class_array.append([1])
    img = sp.imread(path.join(img_folder, img), True)
    img_list.append(img)

img_shape = img_list[0].shape
# # plt.imshow(img_list[0],cmap="gray")


#Resimleri önce satır halinde diziye atıp transpose yaparak sütun haline getirdik
imgs_mtrx = np.array([img.flatten() for img in img_list]).T

#Ortalama resmi bulmak için bütün sütunları toplayarak toplam resim sayısına böldük
#np.sum belirli satır veya sütundaki sayıları toplamamızı sağlar
mean_img = np.sum(imgs_mtrx, axis=1) / img_count

mean_img_2d = mean_img.reshape(img_shape)
# plt.imshow(mean_img_2d, cmap="gray")
# plt.show()

# Toplam resim matrisinden ortalama matrisi çıkararak fark matrisini bulduk
d_arr = imgs_mtrx.T - mean_img
d_array = d_arr.T



# Eigenvector, Eigenvalue değerlerini numpy kütüphanesi yardımı ile hesapladık. PCA
U, s, V = np.linalg.svd(d_array, full_matrices=False)
efaces_array = U

# plt.plot(s)
# plt.show()

#eigenface matrisinin transpozu ile fark matrisini çarparak eigenfacelerin ağırlıklarını hesapladık
weight_array = np.matmul(efaces_array.T,d_array)

#Eigenface ler arasında en yüksek dereceli olan ilk 100 tanesini ve karşılık gelen ağırlıkları tespit ettik
n_eface_array = efaces_array.T[0:57]
n_eface_array = n_eface_array.T[0:57]
n_w_array = weight_array[0:57].T
# n_eface_array = efaces_array


# Normalizasyon
max_column = list()
for i in range(len(n_w_array[0])):
    max = 0
    for j in range(len(n_w_array)):
        if float(n_w_array[j][i]) > max:
            max = float(n_w_array[j][i])
    max_column.append(max)

for i in range(len(n_w_array[0])):
    for j in range(len(n_w_array)):
        n_w_array[j][i] = float(n_w_array[j][i]) / max_column[i]



print(len(n_w_array))
print(len(class_array))
# # print(efaces_array.T.shape)
# # print(n_w_array.shape)
# np.save("eigenface_data/eigenface_weight", n_w_array.T)
np.save("eigenface_data/mean_img", mean_img)
np.save("eigenface_data/max_w_array", max_column)
np.save("eigenface_data/eigenface", efaces_array)
np.save("nn_data/nn_input", n_w_array)
np.save("nn_data/class_array", class_array)

# #Her resmin kendi eigenfaceleri ile ağırlıklarının çarpımının toplamı resmin orjinal görüntüsünü vermektedir.
# recons_imgs = list()
# for c_idx in range(img_count):
#     ri = mean_img + np.dot(efaces_array, weight_array[:, c_idx])
#     recons_imgs.append(ri.reshape(img_shape))
#


# plt.imshow(recons_imgs[30].reshape(50,50),cmap="gray")
# plt.show()
#elde ettiğimiz egainface ve ortalama resmi klasörlere kayıt ettik
# if not path.exists(out_folder): makedirs(out_folder)
# if not path.exists(eface_folder): makedirs(eface_folder)
# for idx, img in enumerate(recons_imgs):
#     sp.imsave(path.join(out_folder, "img_" + str(idx) + ".jpg"), img)
# sp.imsave(path.join(out_folder, "mean.jpg"), mean_img_2d)
#
# for idx in range(n_eface_array.shape[1]):
#     sp.imsave(path.join(eface_folder, "eface" + str(idx) + ".jpg"),
#               n_eface_array[:, idx].reshape(img_shape))

