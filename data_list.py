import numpy as np
import cv2
import os

#Veri setindeki tüm resimlerin yüzlerini tespit edip kırma işlemi yapıyoruz.
#Kırpılan resimler yaş klasörlerine göre dosyalanarak yeniden boyutlanıp kayıt ediliyor.

# face_cascade = cv2.CascadeClassifier('../cascade/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('../cascade/haarcascade_eye.xml')


folder = "../data2"
count = 0
for filename in os.listdir(folder):
    crop_img = []
    if filename is not None:
        f_name = filename.split('_')
        img_s = int(f_name[1])
        if img_s == 0:
            img_age = "_a"
        elif img_s == 1:
            img_age = "_b"
        else:
            img_age = "other-"
        img = cv2.imread(folder+"/"+filename)
        resized_image = cv2.resize(img, (50, 50))
        img_name = "gender_data/"+str(filename[0:-4])+img_age+".jpg"
        cv2.imwrite(img_name, resized_image)
        count += 1