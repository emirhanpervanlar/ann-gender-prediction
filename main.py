import numpy as np
import cv2
from os import listdir, path, makedirs
import scipy.misc as sp
import matplotlib.pyplot as plt
from tkinter import *

def act_sig(x):
    act = 1/(1+np.exp(-1*x))
    return act_step(act)

def act_step(x):
    if x < 0.50:
        return 0
    elif x >= 0.50:
        return 1

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

def predic(img):
    face_img = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_img = img[y:y + h, x:x + w]
        resized_image = cv2.resize(face_img, (50, 50))
        cv2.imwrite("pre_data/img2.jpg", resized_image)
        load_img = sp.imread(path.join("pre_data", "img2.jpg"), True)
        load_img_flat = load_img.flatten().T
        d_arr = load_img_flat.T - mean_img
        d_array = d_arr
        weight_array = np.matmul(efaces_array.T, d_array)[0:57]
        # Normalizasyon
        for i in range(57):
            if (weight_array[i] > max_w_array[i]):
                max_w_array[i] = weight_array[i]
            weight_array[i] = weight_array[i] / max_w_array[i]
        input_data = weight_array.T
        h1_bias = load_h1_bias
        output_bias = load_output_bias
        input_weight = load_input_weight
        h1_weight = load_h1_weight
        h1_out = feedforward(input_data, input_weight, h1_bias[0])
        n_out = feedforward(h1_out, h1_weight, output_bias[0])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 00), 2)
        text = ""
        if (np.array_equal(n_out, [0])):
            text = "Bay"
        elif (np.array_equal(n_out, [1])):
            text = "Bayan"
        cv2.putText(img, text, (x, y + h + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, color=(0, 255, 0),
                    lineType=cv2.LINE_AA)
    cv2.imshow('image', img)
    cv2.waitKey(1)

def img1():
    img = cv2.imread('pre_data/img.jpg')
    predic(img)

def img2():
    img = cv2.imread('pre_data/0x0-2.jpg')
    predic(img)

def img3():
    img = cv2.imread('pre_data/ogrencilere-toplu-dogum-gunu-kutlamasi.jpg')
    predic(img)

def img4():
    img = cv2.imread('pre_data/Friends-Konusu-ve-Karakterleri-620x400.jpg')
    predic(img)

def web_cam():
    face_img = []
    cap = cv2.VideoCapture(0)
    while (True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            resized_image = cv2.resize(face_img, (50, 50))
            cv2.imwrite("pre_data/img2.jpg", resized_image)
            load_img = sp.imread(path.join("pre_data", "img2.jpg"), True)
            load_img_flat = load_img.flatten().T
            d_arr = load_img_flat.T - mean_img
            d_array = d_arr
            weight_array = np.matmul(efaces_array.T, d_array)[0:57]
            for i in range(57):
                if (weight_array[i] > max_w_array[i]):
                    max_w_array[i] = weight_array[i]
                weight_array[i] = weight_array[i] / max_w_array[i]
            input_data = weight_array.T
            h1_bias = load_h1_bias
            output_bias = load_output_bias
            input_weight = load_input_weight
            h1_weight = load_h1_weight
            h1_out = feedforward(input_data, input_weight, h1_bias[0])
            n_out = feedforward(h1_out, h1_weight, output_bias[0])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 00), 2)
            text = ""
            if (np.array_equal(n_out, [0])):
                text = "Bay"
            elif (np.array_equal(n_out, [1])):
                text = "Bayan"
            cv2.putText(frame, text, (x, y + h + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, color=(0, 255, 0),
                        lineType=cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('cascade/haarcascade_eye.xml')
    load_input_weight = np.load("mix_weight/input_weight.npy")
    load_h1_weight = np.load("mix_weight/h1_weight.npy")
    load_h1_bias = np.load("mix_weight/h1_bias.npy")
    load_output_bias = np.load("mix_weight/output_bias.npy")
    mean_img = np.load("eigenface_data/mean_img.npy")
    efaces_array = np.load("eigenface_data/eigenface.npy")
    max_w_array = np.load("eigenface_data/max_w_array.npy")[0:57]

    master = Tk()
    master.title("Yaş Tahmini")
    master.geometry("350x250")

    test_1_btn = Button(text="Resim 1", command=img1, padx=20, pady=5).grid(row=1, column=1, sticky=W)
    test_2_btn = Button(text="Resim 2", command=img2, padx=20, pady=5).grid(row=2, column=1, sticky=W)
    test_3_btn = Button(text="Resim 3", command=img3, padx=20, pady=5).grid(row=3, column=1, sticky=W)
    test_4_btn = Button(text="Resim 4", command=img4, padx=20, pady=5).grid(row=4, column=1, sticky=W)
    cam_btn = Button(text="Kamera", command=web_cam, padx=20, pady=5).grid(row=5, column=1, sticky=W)


    mainloop()