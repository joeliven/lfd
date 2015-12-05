import numpy as np
import os
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import config as c
import time


class emotilines(tk.Tk):
    def __init__(self,parent, time_interval):
        tk.Tk.__init__(self,parent)
        self.parent = parent
        self.x = 0
        self.y = 0
        self.time_interval = time_interval
        self.t = 0
        self.start = None
        self.goal = None
        self.replay_start = None
        self.replay_goal = None
        self.markers = list()
        self.replay_markers = list()

        self.mode_color = "blue"
        self.recording = False
        self.emo_int = 0
        self.mode_int = 0

        self.path = list()
        self.replay_path = list()
        self.initialize()

    def initialize(self):
        self.grid()
        self.geometry("550x750")
        self.grid_columnconfigure(0, minsize=150, weight=1)
        self.grid_columnconfigure(1, minsize=350, weight=1)

        # self.entryVariable = tk.StringVar()
        # self.entry = tk.Entry(self,textvariable=self.entryVariable)
        # self.entry.grid(column=0,row=0,sticky='EW')
        # self.entry.bind("<Return>", self.OnPressEnter)
        # self.entryVariable.set(u"Enter text here.")

        # emo_buttons = list()
        # for i in range(0, len(c.emotions)):
        #     emo_buttons.append(tk.Button(self,text=c.emotions[i],
        #                             command=self.OnButtonClick))


        self.mode = tk.StringVar()
        self.emo = tk.StringVar()
        self.recording_str = tk.StringVar()

        self.mode.set(c.modes[0]) # default value
        self.emo.set(c.emotions[0]) # default value
        self.recording_str.set(c.recording[0]) # default value

        mode_dropdown = tk.OptionMenu(self, self.mode, c.modes[0], c.modes[1], c.modes[2], c.modes[3], command=self.change_mode)
        emo_dropdown = tk.OptionMenu(self, self.emo, c.emotions[0], c.emotions[1], c.emotions[2], c.emotions[3], c.emotions[4], c.emotions[5], command=self.change_emo)
        self.recording_button = tk.Button(self,text=c.record[0],fg=c.recording_color[1])
        self.recording_button.bind('<Button-1>',self.toggle_recording)
        self.clear_canvas_button = tk.Button(self,text=c.clear_canvas)
        self.clear_canvas_button.bind('<Button-1>',self.clear_canvas)
        self.EXP_play_recreates_button = tk.Button(self,text="EXP Play Paths")
        self.EXP_play_recreates_button.bind('<Button-1>',self.EXP_play_recreates)

        self.new_start_button = tk.Button(self,text=c.new_start)
        self.new_start_button.bind('<Button-1>',self.new_start)
        self.new_goal_button = tk.Button(self,text=c.new_goal)
        self.new_goal_button.bind('<Button-1>',self.new_goal)
        self.replay_current_path_button = tk.Button(self,text=c.replay_current_path)
        self.replay_current_path_button.bind('<Button-1>',self.replay_current_path)
        self.save_current_path_button = tk.Button(self,text=c.save_current_path)
        self.save_current_path_button.bind('<Button-1>',self.save_demo_to_file)
        self.open_path_from_file_button = tk.Button(self,text=c.open_path_from_file)
        self.open_path_from_file_button.bind('<Button-1>',self.open_path_from_file)
        self.replay_path_from_file_button = tk.Button(self,text=c.replay_path_from_file)
        self.replay_path_from_file_button.bind('<Button-1>',self.replay_path_from_file)

        mode_dropdown.grid(column=0,row=0, sticky='W,E')
        emo_dropdown.grid(column=0,row=1, sticky='W,E')
        self.recording_button.grid(column=0,row=2, sticky='W,E')
        self.new_start_button.grid(column=0,row=3, sticky='W,E')
        self.new_goal_button.grid(column=1,row=3, sticky='W')
        self.clear_canvas_button.grid(column=0,row=4, sticky='W,E')
        self.EXP_play_recreates_button.grid(column=1,row=4, sticky='W')
        self.replay_current_path_button.grid(column=0,row=5, sticky='W,E')
        self.save_current_path_button.grid(column=1,row=5, sticky='W')
        self.open_path_from_file_button.grid(column=0,row=6, sticky='W,E')
        self.replay_path_from_file_button.grid(column=1,row=6, sticky='W')

        self.mode_label_var = tk.StringVar()
        self.emo_label_var = tk.StringVar()
        self.recording_label_var = tk.StringVar()

        color = "blue"
        mode_label = tk.Label(self,textvariable=self.mode_label_var,
                              anchor="w",fg="white",bg=color)
        mode_label.grid(column=1,row=0,columnspan=1,sticky='EW')
        emo_label = tk.Label(self,textvariable=self.emo_label_var,
                              anchor="w",fg="white",bg=color)
        emo_label.grid(column=1,row=1,columnspan=1,sticky='EW')
        self.recording_label = tk.Label(self,textvariable=self.recording_label_var,
                              anchor="w",fg="white",bg=c.recording_color[0])
        self.recording_label.grid(column=1,row=2,columnspan=1,sticky='EW')

        self.mode_label_var.set(str(c.mode_str) + str(c.modes[0]) )
        self.emo_label_var.set(str(c.emo_str) + "Please Select an Emotion")
        self.recording_label_var.set((c.recording_str) + str(c.recording[0]))

        self.grid_columnconfigure(0,weight=1)
        self.resizable(True,True)
        self.update()
        self.geometry(self.geometry())

        self.canvas_width = 500
        self.canvas_height = 500

        self.message = tk.Label(self, text = "Press and Drag the mouse to draw" )
        self.message.grid(column=0,row=7,columnspan=2)

        self.canvas = tk.Canvas(self,
                   width=self.canvas_width,
                   height=self.canvas_height,
                   bd=3,
                   relief="ridge")

        self.canvas.grid(column=0,row=8,columnspan=2)
        self.canvas.bind( "<B1-Motion>", self.detect_motion )
        self.canvas.bind( "<Button-1>", self.start_recording )
        self.canvas.bind( "<ButtonRelease-1>", self.stop_recording )
        self.canvas.bind("<Button-2>",self.clear_canvas)
        self.canvas.bind("<Button-3>",self.toggle_recording)

        self.time_step()


#########################################################################
#Class Methods#
#########################################################################
    def new_start(self, event):
        start_x = np.random.random_integers(5,self.canvas_width-5)
        start_y = np.random.random_integers(5,self.canvas_height-5)
        if self.start == None:
            self.start = self.canvas.create_rectangle(start_x-5, start_y-5, start_x+5, start_y+5, fill="blue" )
        self.canvas.coords(self.start, start_x-5, start_y-5, start_x+5, start_y+5,)

    def new_goal(self, event):
        goal_x = np.random.random_integers(5,self.canvas_width-5)
        goal_y = np.random.random_integers(5,self.canvas_height-5)
        if self.goal  == None:
            self.goal = self.canvas.create_oval(goal_x-5, goal_y-5, goal_x+5, goal_y+5, fill="green" )
        self.canvas.coords(self.goal, goal_x-5, goal_y-5, goal_x+5, goal_y+5,)


    def start_recording(self, event):
        if self.recording:
            self.t = 0
            self.path.clear()
            if self.start == None:
                self.toggle_recording(self)
                tk.messagebox.showwarning("ERROR!", c.no_start_warning)
            elif self.goal == None:
                self.toggle_recording(self)
                tk.messagebox.showwarning("ERROR!", c.no_goal_warning)
            else:
                start_coords = self.canvas.coords(self.start)
                if not ((event.x >= start_coords[0] and event.x <= start_coords[2]) and (event.y >= start_coords[1] and event.y <= start_coords[3])):
                    self.toggle_recording(self)
                    tk.messagebox.showwarning("ERROR!", c.not_on_start_warning)

    def save_demo_to_file(self, event=None):
        if len(self.path) == 0:
                tk.messagebox.showwarning("WARNING!", c.current_path_empty)
                return
        try:
            f = filedialog.asksaveasfile(mode='w', defaultextension=".csv")
            if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
                return
            f.write("mode:"+str(self.mode.get()))
            f.write("\n")
            f.write("emotion:"+str(self.emo.get()))
            f.write("\n")
            start = self.canvas.coords(self.start)
            f.write("start:")
            for i in range(0,len(start)):
                f.write(str(start[i]))
                if i < len(start) - 1:
                    f.write(",")
            f.write("\n")
            goal = self.canvas.coords(self.goal)
            f.write("goal:")
            for i in range(0,len(goal)):
                f.write(str(goal[i]))
                if i < len(goal) - 1:
                    f.write(",")
            f.write("\n")
            while(self.path[0][0] == 0 and self.path[0][1] == 0):
                self.path.pop(0)
            for point in self.path:
                f.write(str(point[0]))
                f.write(",")
                f.write(str(point[1]))
                f.write("\n")

        finally:
            if f is not None:
                f.close()

    def open_path_from_file(self, event=None):
        try:
            f = filedialog.askopenfile(mode='r', defaultextension=".csv")
            if f is None: # askopenfile return `None` if dialog closed with "cancel".
                return
            mode = (f.readline().split(":")[1]).rstrip()
            self.mode.set(mode)
            self.change_mode(None)

            emo = (f.readline().split(":")[1]).rstrip()
            self.emo.set(emo)
            self.change_emo(None)

            temp = (f.readline().split(":")[1]).rstrip().split(",")
            start_list = list()
            for string in temp:
                start_list.append(float(string))
            if self.replay_start == None:
                self.replay_start = self.canvas.create_rectangle(start_list[0], start_list[1], start_list[2], start_list[3], fill="blue" )
            else:
                self.canvas.coords(self.replay_start, start_list[0], start_list[1], start_list[2], start_list[3])

            temp = f.readline().split(":")[1].split(",")
            goal_list = list()
            for string in temp:
                goal_list.append(float(string))
            if self.replay_goal == None:
                self.replay_goal = self.canvas.create_oval(goal_list[0], goal_list[1], goal_list[2], goal_list[3], fill="green" )
            else:
                self.canvas.coords(self.replay_goal, goal_list[0], goal_list[1], goal_list[2], goal_list[3])

            self.replay_path.clear()
            for line in f:
                x = float(line.split(",")[0].rstrip())
                y = float(line.split(",")[1].rstrip())
                self.replay_path.append((x,y))
            # print("replay_path is: " + str(self.replay_path))
        finally:
            if f is not None:
                f.close()

    def EXP_open_path_from_file(self, fname):
        try:
            f = open(fname, 'r')
            if f is None: # askopenfile return `None` if dialog closed with "cancel".
                print("ERROR, could not load specified file for replay")
                return
            mode = (f.readline().split(":")[1]).rstrip()
            self.mode.set(mode)
            self.change_mode(None)

            emo = (f.readline().split(":")[1]).rstrip()
            self.emo.set("Hidden")
            self.change_emo(None)

            temp = (f.readline().split(":")[1]).rstrip().split(",")
            start_list = list()
            for string in temp:
                start_list.append(float(string))
            if self.replay_start == None:
                self.replay_start = self.canvas.create_rectangle(start_list[0], start_list[1], start_list[2], start_list[3], fill="blue" )
            else:
                self.canvas.coords(self.replay_start, start_list[0], start_list[1], start_list[2], start_list[3])

            temp = f.readline().split(":")[1].split(",")
            goal_list = list()
            for string in temp:
                goal_list.append(float(string))
            if self.replay_goal == None:
                self.replay_goal = self.canvas.create_oval(goal_list[0], goal_list[1], goal_list[2], goal_list[3], fill="green" )
            else:
                self.canvas.coords(self.replay_goal, goal_list[0], goal_list[1], goal_list[2], goal_list[3])

            self.replay_path.clear()
            for line in f:
                x = float(line.split(",")[0].rstrip())
                y = float(line.split(",")[1].rstrip())
                self.replay_path.append((x,y))
            # print("replay_path is: " + str(self.replay_path))
        finally:
            if f is not None:
                f.close()


    def stop_recording(self, event):
        if self.recording:
            goal_coords = self.canvas.coords(self.goal)
            if not ((event.x >= goal_coords[0] and event.x <= goal_coords[2]) and (event.y >= goal_coords[1] and event.y <= goal_coords[3])):
                self.toggle_recording(self)
                tk.messagebox.showwarning("ERROR!", c.not_on_goal_warning)
            else:
                self.toggle_recording(self)
                save = tk.messagebox.askquestion("Save Demo to File?", c.save_demo_to_file)
                # print("save is: " + str(save))
                if save == "yes":
                    self.save_demo_to_file()
                elif save == "no":
                    self.do_clear_canvas()

    def detect_motion(self, event):
        self.x = event.x
        self.y = event.y
        print("detect_motion: x=" + str(self.x) + "\t y=" + str(self.y))

    def paint(self, x, y, color):
       x1, y1 = ( x - 2 ), ( y - 2 )
       x2, y2 = ( x + 2 ), ( y + 2 )
       self.markers.append(self.canvas.create_oval( x1, y1, x2, y2, fill=color ))

    def replay_paint(self, x, y, color):
       x1, y1 = ( x - 2 ), ( y - 2 )
       x2, y2 = ( x + 2 ), ( y + 2 )
       self.replay_markers.append(self.canvas.create_oval( x1, y1, x2, y2, fill=color ))

       # print("event.x=" + str(event.x) + "\t" + "event.y=" + str(event.y))

    def time_step(self):
        x = self.x
        y = self.y
        if self.recording:
            self.t += 1
            self.path.append((x,y))
            print("t: " + str(self.t) + "\tx=" + str(x) + "\ty=" + str(y))
        self.paint(x, y, "blue")
        self.after(self.time_interval, self.time_step)

    def clear_canvas(self, event):
        self.do_clear_canvas()

    def do_clear_canvas(self):
        for marker in self.markers:
            self.canvas.delete(marker)
        self.markers.clear()
        for marker in self.replay_markers:
            self.canvas.delete(marker)
        self.replay_markers.clear()
        # if self.replay_start is not None:
        #     self.canvas.delete(self.replay_start)
        # if self.replay_goal is not None:
        #     self.canvas.delete(self.replay_goal)


    def change_emo(self, event):
        self.emo_label_var.set( c.emo_str + self.emo.get() )
        for i in range(0, len(c.emotions)):
            if self.emo.get() == c.emotions[i]:
                self.emo_int = i
    def change_mode(self, event):
        self.mode_label_var.set( c.mode_str + self.mode.get() )
        for i in range(0, len(c.modes)):
            if self.mode.get() == c.modes[i]:
                self.mode_int = i

    def toggle_recording(self, event):
        # print(event)
        # print(self.recording_str.get())
        if self.recording == False:
            self.recording_label_var.set( c.recording_str + c.recording[1] )
            self.recording_button.config(text=c.record[1],fg=c.recording_color[2])
            self.recording_label.config(bg=c.recording_color[1])
            self.recording = True
        elif self.recording == True:
            self.recording_label_var.set( c.recording_str + c.recording[0] )
            self.recording_button.config(text=c.record[0],fg=c.recording_color[1])
            self.recording_label.config(bg=c.recording_color[0])
            self.recording = False
        else:
            print("ERROR!!!")

    def replay_current_path(self, event=None):
        if len(self.path) == 0:
            tk.messagebox.showwarning("WARNING!", c.current_path_empty)
            return
        # print(str(self.path))
        self.do_replay_current(0,"red")

    def replay_path_from_file(self, event=None):
        if len(self.replay_path) == 0:
            tk.messagebox.showwarning("WARNING!", c.path_from_file_empty)
            return
        # print(str(self.path))
        self.do_replay_file(0,"red")

    def EXP_replay_path_from_file(self, color, event=None):
        if len(self.replay_path) == 0:
            tk.messagebox.showwarning("WARNING!", c.path_from_file_empty)
            return
        # print(str(self.path))
        self.do_replay_file(0, color)


    def do_replay_current(self, idx, color):
        if idx >= len(self.path):
            return
        else:
            x = (self.path[idx])[0]
            y = (self.path[idx])[1]
            self.replay_paint(x, y, color)
            # self.do_replay(idx+1,color)
            # self.after(int(self.time_interval/10), self.do_replay, idx+1, color)
            self.after(int(self.time_interval), self.do_replay_current, idx+1, color)

    def do_replay_file(self, idx, color):
        if idx >= len(self.replay_path):
            return
        else:
            x = (self.replay_path[idx])[0]
            y = (self.replay_path[idx])[1]
            self.replay_paint(x, y, color)
            # self.do_replay(idx+1,color)
            # self.after(int(self.time_interval/10), self.do_replay, idx+1, color)
            self.after(int(self.time_interval), self.do_replay_file, idx+1, color)

    def EXP_play_recreates(self, event):
        try:
            f = filedialog.askopenfile(mode='r', defaultextension=".csv")
            if f is None: # askopenfile return `None` if dialog closed with "cancel".
                return
            path_filenames = list()
            for line in f:
                fname = line.rstrip()
                print("fname is: " + str(fname))
                path_filenames.append(line.rstrip())
        finally:
            f.close()

        print("after finally block")
        colors = ["red", "blue", "green"]
        c = 0
        for path_file in path_filenames:
            self.EXP_open_path_from_file(path_file)
            self.EXP_replay_path_from_file(colors[c%3])
            time.sleep(2000)
            c += 1
            if c%3 == 0:
                self.do_clear_canvas()



if __name__ == "__main__":
    app = emotilines(None, c.time_interval)
    app.title('EmotiLines')
    app.mainloop()

