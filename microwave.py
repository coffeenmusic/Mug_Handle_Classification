import cv2
from tkinter import *
from PIL import ImageTk, Image
from time import sleep
from time import gmtime, strftime
import os

class Microwave(object):
    def __init__(self):
        # Camera------------------------------------        
        self.cam = cv2.VideoCapture(0)
        if not(self.cam.isOpened()):
            print('Error opening camera')
        
        self.capture_delay = 0.5 # seconds
        self.capture_delay_ms = int(self.capture_delay*1000)
        
        no_errors, img = self.cam.read()
        if type(img) == None:
            print("Image NoneType")
            return (False, self.no_img)
        if no_errors:
            self.rotated_img = self.rotate_img(img, 180)
        
        # GUI----------------------------------------
        self.root = Tk()        
        self.btnRun_lbl = "Run"
        self.btnStop_lbl = "Stop"
        self.btnTest_lbl = "Test Network"
        self.btnFacingDoor_lbl = "Facing Door"
        
        no_img = cv2.imread("no_img.jpg", 0)
        no_img = Image.fromarray(no_img)
        self.no_img = ImageTk.PhotoImage(no_img)
        
        # Functionality------------------------------
        self.flg_run = False
        self.flg_handle = False
        self.flg_test = False
        self.img_path = "Images/"
        
        self.img_index = self.get_img_index()
        
    def get_img_index(self):
        subdirs = os.listdir(self.img_path)
        classes = [each for each in subdirs if os.path.isdir(self.img_path + each)]
        return max([int(f[:-4]) for c in classes for f in next(os.walk(self.img_path + c))[2] if f.endswith(".jpg")]) + 1
        
    def facing_door(self, event):
        self.flg_handle = True
        print("Facing Door")
        
    def not_facing_door(self, event):
        self.flg_handle = False
        print("Not Facing Door")
        
    def launch(self):
        btnRun = Button(self.root, text=self.btnRun_lbl, command=self.btn_run_press)
        btnRun.grid(row=2, column=0, sticky=N+S+E+W)

        btnStop = Button(self.root, text=self.btnStop_lbl, command=self.run_stop)
        btnStop.grid(row=2, column=1, sticky=N+S+E+W)
        
        btnTest = Button(self.root, text=self.btnTest_lbl, command=self.run_test)
        btnTest.grid(row=2, column=2, sticky=N+S+E+W)

        btnFacingDoor = Button(self.root, text=self.btnFacingDoor_lbl)
        btnFacingDoor.grid(row=2, column=2, sticky=N+S+E+W)
        btnFacingDoor.bind('<Button-1>', self.facing_door)
        btnFacingDoor.bind('<ButtonRelease-1>', self.not_facing_door)
        
        self.display_img = Label(image=self.no_img)
        self.display_img.grid(row=0, columnspan=3)

        self.root.mainloop()
        
    def rotate_img(self, img, angle):
        rows, cols, _ = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        return cv2.warpAffine(img, M, (cols,rows))
        
    def btn_run_press(self):
        self.run_start()
        
    def run_start(self):
        print("Run Start")
        self.flg_run = True
        
        # Instantiate Callback Timer
        self.root.after(self.capture_delay_ms, self.update_img)
            
    def run_stop(self):
        print("Run Stop")
        self.flg_run = False
        self.flg_test = False
        
    def run_test(self):
        print("Run Test")
        self.flg_test = True
        
    def cam_capture(self):
        no_errors, img = self.cam.read()
        if type(img) == None:
            print("Image NoneType")
            return (False, self.no_img)
        if no_errors:
            self.rotated_img = self.rotate_img(img, 180)
            camera = cv2.cvtColor(self.rotated_img, cv2.COLOR_BGR2RGB)
            camera = Image.fromarray(camera)
            camera = ImageTk.PhotoImage(camera)
            return (True, camera)
        else:
            print("Image Errors")
            return (False, self.no_img)
        
    def update_img(self):
        if self.flg_run == False:
            return
        
        success, camera = self.cam_capture()
        self.display_img.configure(image=camera)
        self.display_img.image = camera
        if success:
            print("Success")
            filename = str(self.img_index) +".jpg"
            path = self.img_path + str(self.flg_handle) + "/" + filename
            cv2.imwrite(path, self.rotated_img)
            self.img_index += 1
        
        # Re-Instantiate Callback Timer
        self.root.after(self.capture_delay_ms, self.update_img)