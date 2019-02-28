import os
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()


def load_model(root):
    root.withdraw()
    print("Initializing Dialogue... \nPlease select a model file.")
    tk_filenames = filedialog.askopenfilenames(initialdir= os.getcwd(), filetypes = [('Model', '.h5'), ('all files', '*.*'),], title='Please select one or more files', multiple = False)
    filenames = list(tk_filenames)
    return filenames



def get_filenames(root):
    root.withdraw()
    print("Initializing Dialogue... \nPlease select a file.")
    tk_filenames = filedialog.askopenfilenames(initialdir=os.getcwd(), filetypes = [('Images', '.jpg'), ('all files', '*.*'),], title='Please select one or more files')
    filenames = list(tk_filenames)
    filenames = root.tk.splitlist(filenames)
    return filenames


images = get_filenames(root)

for image in images:
    print("Image", image)

