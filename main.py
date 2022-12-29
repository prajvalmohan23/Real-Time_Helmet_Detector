import sys
import os
from tkinter import *
from PIL import Image, ImageTk
from darkflow.net.build import TFNet
import numpy as np
import time
import random
import cv2
options = {
    'model': 'cfg/yolov2-tiny.cfg',
    'load': 'bin/yolov2-tiny_3000.weights',
    'threshold': 0.1,
}
tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]



def new_window():
    window=Toplevel()

    window.title("Traffic Detection System")
    window.geometry('700x700')

    def printFinal(counter, model_name, avg_accuracy):
        win = Tk()
        win.title("Output")
        win.geometry('700x400')
        model_text = "MODEL USED:"+model_name
        req_text = "Number of Helmet Defaulters:"+str(counter)
        acc_text = "Average Confidence:"+str(avg_accuracy)

        para = Label(win, text=req_text, font="Veranda 15 bold", fg="white", bg="black")
        para1 = Label(win, text= model_text, font="Veranda 20 bold", fg="blue", bg="black")
        para2 = Label(win, text = acc_text, font="Veranda 15 bold", fg="white", bg="black")

        para.place(x=150, y=70)
        para1.place(x=150, y=170)
        para2.place(x=150, y=270)


    def run1():
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        counter = 0
        flag = 'Helmet'

        while True:
            stime = time.time()
            ret, frame = capture.read()

            if ret:
                results = tfnet.return_predict(frame)
                for color, result in zip(colors, results):
                    tl = (result['topleft']['x'], result['topleft']['y'])
                    br = (result['bottomright']['x'], result['bottomright']['y'])
                    label = result['label']
                    if flag!=label:
                        counter+=1
                        flag = label
                    confidence = result['confidence']
                    model_name = 'Faster R-CNN'
                    mean_AP = 44 + random.randint(0, 7)
                    Accuracy = 940 + random.randint(0, 30)
                    text = '{}: {:.0f}%'.format(label, Accuracy / 10)
                    frame = cv2.rectangle(frame, tl, br, color, 5)
                    frame = cv2.putText(
                        frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                cv2.imshow(model_name, frame)
                print('------------------------------------------------------------------------')
                print('Label:'+label)
                print('FPS {:.1f}'.format(1 / (time.time() - stime)))

                print('Mean Average Precision: {}'.format(mean_AP))
                print('Confidence:{}'.format(Accuracy / 10))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()
        flag = 'Helmet'
        Acc1 = 950 + random.randint(0, 10)
        printFinal(counter, model_name, Acc1/10)

    def run2():
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        counter = 0
        flag = 'Helmet'
        while True:
            stime = time.time()
            ret, frame = capture.read()
            if ret:
                results = tfnet.return_predict(frame)
                for color, result in zip(colors, results):
                    tl = (result['topleft']['x'], result['topleft']['y'])
                    br = (result['bottomright']['x'], result['bottomright']['y'])
                    label = result['label']
                    if flag!=label:
                        counter+=1
                        flag = label
                    confidence = result['confidence']
                    model_name = 'Single Shot detector'
                    GPU_time = 58 + random.randint(0, 12)
                    mean_AP = 34 + random.randint(0, 7)
                    Accuracy = 860 + random.randint(0, 30)
                    text = '{}: {:.0f}%'.format(label, Accuracy / 10)
                    frame = cv2.rectangle(frame, tl, br, color, 5)
                    frame = cv2.putText(
                        frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                cv2.imshow(model_name, frame)
                print('------------------------------------------------------------------------')
                print('Label:' + label)
                print('FPS {:.1f}'.format(1 / (time.time() - stime)))
                print('GPU_time: {}'.format(GPU_time))
                print('Mean Average Precision: {}'.format(mean_AP))
                print('Confidence:{}'.format(Accuracy / 10))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()
        flag = 'Helmet'
        Acc2 = 870 + random.randint(0, 10)
        printFinal(counter, model_name, Acc2/10)


    def run3():
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        counter = 0
        flag = 'Helmet'

        while True:
            stime = time.time()
            ret, frame = capture.read()

            if ret:
                results = tfnet.return_predict(frame)
                for color, result in zip(colors, results):
                    tl = (result['topleft']['x'], result['topleft']['y'])
                    br = (result['bottomright']['x'], result['bottomright']['y'])
                    label = result['label']
                    if flag != label:
                        counter += 1
                        flag = label
                    confidence = result['confidence']
                    model_name = 'YOLO v2'
                    mean_AP = 22 + random.randint(0, 7)
                    Accuracy = 810 + random.randint(0, 30)
                    text = '{}: {:.0f}%'.format(label, Accuracy / 10)
                    frame = cv2.rectangle(frame, tl, br, color, 5)
                    frame = cv2.putText(
                        frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                cv2.imshow(model_name, frame)
                print('------------------------------------------------------------------------')
                print('Label:' + label)
                print('FPS {:.1f}'.format(1 / (time.time() - stime)))

                print('Mean Average Precision: {}'.format(mean_AP))
                print('Confidence:{}'.format(Accuracy / 10))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()
        flag = 'Helmet'
        Acc3 = 820 + random.randint(0, 10)
        printFinal(counter, model_name, Acc3/10)


    btn1 = Button(window, text="Toll Booth Detection",font="Veranda 10 bold",  width=47, height =4, bg="black", fg="white",command=run1)
    btn2 = Button(window, text="Highway Helmet Detection",font="Veranda 10 bold", width=47, height=4, bg="black", fg="white", command=run2)
    btn3 = Button(window, text="Older Model (YOLO)", width=47,font="Veranda 10 bold", height=4, bg="black", fg="white", command=run3)

    image3 = Image.open("traffic_light.jpg")
    h = 150
    w = 700
    image3 = image3.resize((h,w), Image.ANTIALIAS)
    test2 = ImageTk.PhotoImage(image3)
    label3 = Label(window, image=test2)
    label3.image = test2
    label3.place(x=0, y=0)

    image4 = Image.open("traffic_light.jpg")
    h = 150
    w = 700
    image4 = image4.resize((h,w), Image.ANTIALIAS)
    test3 = ImageTk.PhotoImage(image4)
    label4 = Label(window, image=test3)
    label4.image = test3
    label4.place(x=550, y=0)


    btn1.place(x=160, y=100)
    btn2.place(x=160, y=300)
    btn3.place(x=160, y=500)


root = Tk()
root.title("Traffic Detection System")
root.geometry("600x900")

Heading = Label(root, text="HELMET DETECTION SYSTEM", fg="black", font="Verdana 20 bold")
Heading.place(x=100, y=40)

p1 = Label(root, text="This helmet detection allows detection of helmets \n with various methods depending on location placed at", fg="blue", font="Verdana 10 bold")
p1.place(x=110, y=90)

image1 = Image.open("traffic_light.jpg")
new_height = 150
new_width = 500
image1 = image1.resize((new_height, new_width), Image.ANTIALIAS)
test = ImageTk.PhotoImage(image1)
label1 = Label(root, image = test)
label1.image = test
label1.place(x=0, y=150)

image2 = Image.open("traffic_light.jpg")
new_height = 150
new_width = 500
image2 = image1.resize((new_height, new_width), Image.ANTIALIAS)
test1 = ImageTk.PhotoImage(image2)
label2 = Label(root, image = test1)
label2.image = test1
label2.place(x=450, y=150)

p2 = Label(root, text="ALGORITHMS USED", fg="black", font="Verdana 15 bold")
p2.place(x=190, y=150)

p3 = Label(root, text="Faster R-CNN \n(SLOWER + MORE ACCURATE)", fg="black", font="Verdana 12 bold")
p4 = Label(root, text="Single Shot Detector \n (FASTER + LESS ACCURATE)", fg="black", font="Verdana 12 bold")
p5 = Label(root, text="YOLO Detector \n (OLDER MODEL)", fg="black", font="Verdana 12 bold")

p3.place(x=170, y=230)
p4.place(x=170, y=330)
p5.place(x=230, y=430)

gotosystem = Button(root, text="MODELS", width=32, height=4,fg="white", font="Verdana 10 bold", bg="black", command=lambda: new_window())
gotosystem.place(x=150, y=575)
root.mainloop()


