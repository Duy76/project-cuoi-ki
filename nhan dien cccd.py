import torch
import cv2
import os
import time
import numpy as np
import tkinter.messagebox
from modules.text_recognition.vietocr.tool.predictor import Predictor
from modules.text_recognition.vietocr.tool.config import Cfg
from modules.detect_word import OCR
from modules.crop_image import CropImg
from id_card_aligment import preprocessing
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
def vietnamese_ocr():
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = './models/transformerocr.pth'
    config['cnn']['pretrained'] = False
    config['device'] = 'cpu'
    config['predictor']['beamsearch'] = False
    detector = Predictor(config)
    return detector
IMG_SIZE = 450
model_crop = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov5_l6_cccd.pt')
detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/information_yolov5l.pt')
nlp_model = vietnamese_ocr()
def exitt():
    exit()
def select_image():
    global panelA, panelB
    path = filedialog.askopenfilename()
    if len(path) > 0:
        image_process, image_origin = preprocessing(path, model_crop, 640, debug=True)
        detect_id_card = detect_model(image_process, IMG_SIZE)
        bbox = detect_id_card.pandas().xyxy[0].to_dict(orient="records")
        threshold = 0.5
        for j in range(len(detect_id_card.xyxy)):
            field_dict = OCR(detect_id_card.xyxy[j], detect_id_card.pandas().xyxy[j], image_process, nlp_model)
            print(field_dict)
            for result in bbox:
                print(result['class'])
                con = result['confidence']
                x1 = int(result['xmin'])
                y1 = int(result['ymin'])
                x2 = int(result['xmax'])
                y2 = int(result['ymax'])
                if result['class'] == 0:
                    crop = image_process[y1:y2, x1:x2]
                    cv2.imwrite('static/downloads/' + 'result_2' + '.png', crop)
                if con > threshold:
                    cv2.rectangle(image_process, (x1, y1), (x2, y2), (0, 0, 255), 2)
            ChatLog.config(state=NORMAL)
            ChatLog.insert(END, "qr: " + 'True' + '\n'
                                 "id:" + field_dict['id'],
                                 "name" + field_dict['name'],
                                 "birth" + field_dict['birth'],
                                 "gender" + field_dict['gender'],
                                 "country" + field_dict['country'],
                                 "home" + field_dict['home'],
                                 "address" + field_dict['add'],
                                 "valid until" + field_dict['valid'])
            ChatLog.config(state=DISABLED)
            ChatLog.yview(END)
        image_origin = Image.fromarray(image_origin)
        image_process = Image.fromarray(image_process)
        image_origin = ImageTk.PhotoImage(image_origin)
        image_process = ImageTk.PhotoImage(image_process)
        if panelA is None or panelB is None:
            # the first panel will store our original image
            panelA = Label(image=image_origin)
            panelA.image = image_origin
            panelA.pack(side="left", padx=10, pady=10)
            # while the second panel will store the edge map
            panelB = Label(image=image_process)
            panelB.image = image_process
            panelB.pack(side="right", padx=10, pady=10)
            panelB.place(x=1000, y=10)
            ChatLog.config(state=NORMAL)
            ChatLog.insert(END, "qr: " + 'True' + '\n\n'
                           "id: " + field_dict['id'] + '\n\n'
                           "name: " + field_dict['name'] + '\n\n'
                           "birth: " + field_dict['birth'] + '\n\n'
                           "gender: " + field_dict['gender'] + '\n\n'
                           "country: " + field_dict['country'] + '\n\n'
                           "home: " + field_dict['home'] + '\n\n'
                           "address: " + field_dict['add'] + '\n\n'
                           "valid until: " + field_dict['valid'])
            ChatLog.config(state=DISABLED)
            ChatLog.yview(END)
        else:
            # update the pannels
            panelA.configure(image=image_origin)
            panelB.configure(image=image_process)
            panelA.image = image_origin
            panelB.image = image_process
panelA = None
panelB = None
root = Tk()
#root.geometry('900x500')
btn = Button(root, text="Chọn ảnh cần xuất thông tin", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn.place(x=20, y=100)
#root = Tk()00
#root.geometry('900x500')
#btn = Button(root, text="Select an image", command=select_image)
33
#btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
ChatLog = Text(root, bd=0, bg="red", height="10", width="10", font="Arial",)
ChatLog.place(x=16,y=16, height=10, width=10)
ChatLog.config(state=DISABLED)
ChatLog.place(x=800, y=330)
root.mainloop()

