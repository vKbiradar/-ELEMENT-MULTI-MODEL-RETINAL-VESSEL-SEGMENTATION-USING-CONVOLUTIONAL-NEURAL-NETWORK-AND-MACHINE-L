# -*- coding: utf-8 -*-ss
"""
Created on Tue May  4 17:28:41 2021

@author: user
"""

import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox as ms
import cv2
import sqlite3
import os
import numpy as np
import time


root=tk.Tk()

root.title("Retinal Vessels Segmentation System")
w,h = root.winfo_screenwidth(),root.winfo_screenheight()

bg = Image.open(r"bh3.jpg")
#bg.resize((1366,500),Image.ANTIALIAS)
print(w,h)
bg_img = ImageTk.PhotoImage(bg)
bg_lbl = tk.Label(root,image=bg_img)
bg_lbl.place(x=0,y=93)
#, relwidth=1, relheight=1)



w = tk.Label(root, text="Retinal Vessels Segmentation System",width=35,background="#800517",foreground="white",height=2,font=("Times new roman",19,"bold"))
w.place(x=0,y=0)



w,h = root.winfo_screenwidth(),root.winfo_screenheight()
root.geometry("%dx%d+0+0"%(w,h))
root.configure(background="#800517")


#from tkinter import messagebox as ms


def Login():
    from subprocess import call
    call(["python","login.py"])
def Register():
    from subprocess import call
    call(["python","registration.py"])


wlcm=tk.Label(root,text="......Welcome to Retinal Vessels Segmentation System ......",width=110,height=3,background="#800517",foreground="white",font=("Times new roman",19,"bold"))
wlcm.place(x=0,y=700)




d2=tk.Button(root,text="Login",command=Login,width=7,height=2,bd=0,background="#800517",foreground="white",font=("times new roman",17,"bold"))
d2.place(x=600,y=0)


d3=tk.Button(root,text="Register",command=Register,width=8,height=2,bd=0,background="#800517",foreground="white",font=("times new roman",17,"bold"))
d3.place(x=700,y=0)



root.mainloop()
