import DataProcessing
import model
import os
import numpy as np
import tkinter as tk
from tkinter import Button
from tkinter import filedialog
from tkinter import Label


base = tk.Tk()
base.title('ALRS')
base.configure(background='#5b717a')
base.geometry('500x350')

titleLab = Label(base, text='Automated Lip Reading Software', font=('Calibri', 25), pady=20, padx=25, borderwidth=5, relief='solid', bg='#adb6ba')
titleLab.pack()


def loadvideo():
    global fileCompletePath
    global fileName
    fileCompletePath = filedialog.askopenfilename(initialdir='./', title='Select Video', filetypes=[("mp4 files", "*.mp4")])
    fileName = DataProcessingFinal.pointsExtraction(fileCompletePath)


open = Button(base, text='Load Video', command=loadvideo, border=5, font=('Calibri', 20), bg='#adb6ba')
open.place(x=0, y=90)


def loadweights():
    model.loadWeights()


loadModel = Button(base, text='Compile Model', command=loadweights, border=5, font=('Calibri', 20), bg='#adb6ba')
loadModel.place(x=0, y=155)

predicted = Label(base, width=10, bg='#adb6ba', font=('Calibri', 25), borderwidth=5, relief='solid')

def makeprediction():
    result = model.prediction(fileName)
    print(result)
    result1 = np.amax(result[0])
    result = np.where(result[0] == result1)

    if result[0] == 0:
        predicted.configure(text='About')
        predicted.place(x=298, y=182.5)
    elif result[0] == 1:
        predicted.configure(text='Banks')
        predicted.place(x=298, y=182.5)
    elif result[0] == 2:
        predicted.configure(text='Called')
        predicted.place(x=298, y=182.5)

prediction = Button(base, text='Make Prediction', command=makeprediction, border=5, font=('Calibri', 20), bg='#adb6ba')
prediction.place(x=0, y=220)

def  video():
    os.system(fileCompletePath)


playVideo = Button(base, text='Play Video', command=video, border=5, font=('Calibri', 20), bg='#adb6ba')
playVideo.place(x=0, y=285)

base.resizable(0,0)
base.mainloop()



