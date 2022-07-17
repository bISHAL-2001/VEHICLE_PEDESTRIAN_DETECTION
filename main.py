from Detect import *
import os
import cv2
import tkinter as tk
from tkinter import messagebox  # provides a template base class as well as a variety of convenience methods for commonly used configurations
from tkinter import ttk  # provides access to the Tk themed widget set
from tkinter import filedialog as fd  # provides classes and factory functions for creating file/directory selection windows  # provides a template base class as well as a variety of convenience methods for commonly used configurations


def message_box(time_set):
    win = tk.Tk()
    win.withdraw()
    messagebox.showinfo("Info", f"Time span set to {time_set} secs. for data capture!")
    win.destroy()


def input_window(seconds):

    temp = seconds
    minutes: int = 0
    hours: int = 0
    if seconds > 60:
        minutes = seconds // 60
        seconds = seconds % 60
        if minutes > 60:
            hours = minutes // 60
            minutes = minutes % 60

    root = tk.Tk()
    root.title("Time Input")
    root.geometry("280x110")
    root.resizable(False, False)

    time_var = tk.StringVar()  # captures and stores the input from the user via the inout window
    option = ["secs", "mins", "hrs"]  # Options for the drop-down list
    time_label = tk.Label(root, text='TIME', font=('', 12, 'bold'))  # Label for the entry widget

    # For proper timespan of the video message
    time_stamp: str
    if minutes and hours:
        time_stamp = f"Time of the video extends to {hours} h {minutes} m {seconds} s!"
    elif minutes:
        time_stamp = f"Time of the video extends to {minutes} m {seconds} s!"
    else:
        time_stamp = f"Time of the video extends to {seconds} secs!"

    time_of_video_label = tk.Label(root, text=time_stamp, fg="#FF0000", font=("Helvetica", 10, "bold"))  # Label for the timespan message
    time_entry = tk.Entry(root, textvariable=time_var, font=('calibre', 10, 'normal'))  # Time entry widget

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

    seconds = temp

    # For submit button action
    def submit():
        nonlocal time_elapsed
        time_1 = time_var.get()
        unit = clicked.get()
        time_var.set("")

        # To convert time elapsed to secs & checking if the time is in int and the limit is within the time of the video
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

    # Limiting the calculation time for frames of at least 1 second
    if time_elapsed == 0:
        exit(1)
    return time_elapsed


# For selecting video path
def select_video_path():

    # Window for the select file button
    root = tk.Tk()
    root.title('Select File')
    root.resizable(False, False)
    root.geometry('250x100')

    # stores the video path
    filename: str = ""

    # function for selecting the video from the file system
    def select_file():
        # classifying type of files between video types and all types
        filetypes = (
            ("Video Files", '*.mp4'),
            ("Video Files", '.avi'),
            ("Video Files", '.mkv'),
            ("Video Files", '.mov'),
            ('All Files', '*.*')
        )

        # For storing the file path within the nested function
        nonlocal filename

        # function returns the file name that you selected in <str> variable filename
        filename = fd.askopenfilename(
            title='Select Video',
            initialdir='/',
            filetypes=filetypes)

        # If the file selected is of the video types defined
        if ".mp4" in filename or ".avi" in filename or ".mkv" in filename or ".mov" in filename:
            messagebox.showinfo('Selected File', f"{filename}")  # shows the selected file path
        root.destroy()

    # open button
    open_button = ttk.Button(root, text='Select from Files', command=select_file)
    open_button.pack(expand=True)

    # run the application
    root.mainloop()

    # If no files is selected or the select window is closed then clos the application with a no file error
    if filename == "":
        w = tk.Tk()
        w.withdraw()
        messagebox.showerror("Error", "No file selected!")
        w.destroy()
        exit(1)

    # # If the file selected file is of the defined video type return the file path
    elif filename[-3:] == "mp4" or filename[-3:] == "mov" or filename[-3:] == "mkv" or filename[-3:] == "avi":
        return filename

    # If a wrong files is selected then close the application with a File format error
    else:
        w = tk.Tk()
        w.withdraw()
        messagebox.showerror("File Type Error", "ACCEPTED CODECS : mp4, mov, mkv, avi")
        w.destroy()
        exit(1)


def main():
    # For object detection and object tracking in either video or live footage
    video_path = select_video_path()

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

    time_elapsed = input_window(seconds)  # stores the duration within which the data is captured

    detector = Detector(video_path, config_path, model_path, classes_path)  # initializing detector object of the class Detect
    message_box(time_elapsed)  # Prompt for the user to show the length of the video capture selected
    detector.onVideo(time_elapsed)  # invoking onVideo function of Detect class with the time set for video capture


if __name__ == '__main__':
    main()
