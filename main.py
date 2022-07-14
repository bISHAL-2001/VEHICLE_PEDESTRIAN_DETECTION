from Detect import *
import os
import cv2
import tkinter as tk
from tkinter import messagebox


def message_box(time_set):
    win = tk.Tk()
    win.withdraw()
    messagebox.showinfo("Info", f"Time span set to {time_set} secs. for data capture!")
    win.destroy()


def input_window(seconds):
    root = tk.Tk()
    root.title("Time Input")
    root.geometry("280x110")
    root.resizable(False, False)

    time_var = tk.StringVar()
    option = ["secs", "mins", "hrs"]
    time_label = tk.Label(root, text='TIME', font=('', 12, 'bold'))
    time_of_video_label = tk.Label(root, text=f"Time of the video extends to {seconds} secs!", fg="#FF0000", font=("Helvetica", 10, "bold"))
    time_entry = tk.Entry(root, textvariable=time_var, font=('calibre', 10, 'normal'))

    clicked = tk.StringVar(root)
    clicked.set("unit")
    drop = tk.OptionMenu(root, clicked, *option)
    time_elapsed = 0

    time_label.place(x=4, y=5)
    time_of_video_label.place(x=19, y=40)
    time_entry.place(x=50, y=6.5)
    drop.place(x=202.5, y=4)
    drop.config(width=4)

    sub_btn = tk.Button(root, text='Submit', width=36, command=lambda: submit())
    sub_btn.place(x=8, y=70)

    def submit():
        nonlocal time_elapsed
        time_1 = time_var.get()
        unit = clicked.get()
        time_var.set("")

        if time_1 != "" and "1" <= time_1[0] <= "9" and unit != "unit" and int(time_1) <= seconds:
            time_elapsed = int(time_1)
            if unit == "secs":
                time_elapsed = time_elapsed
            elif unit == "mins":
                time_elapsed = time_elapsed * 60
            else:
                time_elapsed = time_elapsed * 3600

            root.destroy()

    root.mainloop()
    if time_elapsed == 0:
        time_elapsed = seconds
    return time_elapsed


def main():
    # For object detection and object tracking in either video or live footage
    video_path = "Test_Video2.avi"

    # For Model Configuration
    config_path = os.path.join("model_data", "config.pbtxt")

    # For the model path
    model_path = os.path.join("model_data", "model.pb")

    # For the classes trained in the trained model
    classes_path = os.path.join("model_data", "classes.names")

    # calculate duration of the video
    data = cv2.VideoCapture(video_path)
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(data.get(cv2.CAP_PROP_FPS))
    seconds = int(frames / fps)

    time_elapsed = input_window(seconds)

    detector = Detector(video_path, config_path, model_path, classes_path)
    message_box(time_elapsed)
    detector.onVideo(time_elapsed)


if __name__ == '__main__':
    main()
