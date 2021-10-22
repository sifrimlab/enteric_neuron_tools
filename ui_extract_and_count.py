import sys
import os
import time
import analysis
import argparse
import custom_io
import processing
import numpy as np
import aicspylibczi
import tkinter as Tk
from skimage import io
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename, askdirectory


# global bool because tkinter commands suck
maxIP = True

def runApp():
    window = Tk.Tk()
    window.title("Count enteric neurons")
    window.geometry('900x200')
    window.option_add('*Font', 'Helvetica 16')

    def openImage():
        global czi_path
        czi_path = askopenfilename(title="Select czi file",filetypes=[("image", ".czi")])
        fileLabel.config(text=czi_path)
        fileButton.config(fg="green")
    def openDirectory():
        global out_dir
        out_dir = askdirectory()
        outputDirLabel.config(text=out_dir)
        outputDirButton.config(fg="green")


    fileButton = Tk.Button(window,fg="red", text="Select your czi file: ", command=openImage)
    fileLabel=Tk.Label(window, text="No file selected")
    fileButton.grid(row=0, sticky="W")
    fileLabel.grid(row=0, column=1, sticky="E")

    # output directory
    outputDirButton = Tk.Button(window,fg="red", text="Select your output directory: ", command=openDirectory)
    outputDirLabel=Tk.Label(window, text="No directory selected")
    outputDirButton.grid(row=1, sticky="W")
    outputDirLabel.grid(row=1, column=1, sticky="E")


    # Fill in the channel number
    channelLabel=Tk.Label(window,text="Channel index of marker of interest:")
    channelLabel.grid(row=2, sticky="W")
    channelSpinBox = Tk.Spinbox(from_=0,to=4)
    # channelSpinBox.pack()
    channelSpinBox.grid(row=2, column=1, sticky="E")

    maxIPLabel = Tk.Label(window,text="take maximum intensity projections:")
    maxIPLabel.grid(row=6, sticky="W")

    # Check if maxIP needs to be taken, if not create an extra label to fill in the channel
    def switch():
        global maxIP
        global zstackLabel
        global zstackSpinBox
        # Determine is on or off
        if maxIP:
            on_button.config(image = off)
            maxIP = False
            # Fill in the channel number
            zstackLabel=Tk.Label(window,text="Z-stack index to be used.")
            zstackLabel.grid(row=7, sticky="W")
            zstackSpinBox = Tk.Spinbox(from_=0, to=50)
            zstackSpinBox.grid(row=7, column=1, sticky="E")

        else:
            on_button.config(image = on)
            maxIP = True
            zstackLabel.destroy()
            zstackSpinBox.destroy()

        # Define Our Images
    on = Tk.PhotoImage(file = "imgs/on.png")
    off = Tk.PhotoImage(file = "imgs/off.png")

    # Create A Button for maxIP
    on_button = Tk.Button(window, image = on, bd = 0,command = switch)
    on_button.grid(row=6, column=1, sticky="E")

    # Entries for parameters
    pixelDensityLabel=Tk.Label(window,text="Pixel density (in pixel/micrometer):")
    pixelDensityLabel.grid(row=3, sticky="W")
    pixelDensityEntry=Tk.Entry(window, width=10)
    pixelDensityEntry.grid(row=3, column=1, sticky="E")
    pixelDensityEntry.insert(0, 3.2)

    sigmaLabel=Tk.Label(window,text="Sigma parameter that controls background smoothing:")
    sigmaLabel.grid(row=4, sticky="W")
    sigmaEntry=Tk.Entry(window, width=10)
    sigmaEntry.grid(row=4, column=1, sticky="E")
    sigmaEntry.insert(0, 7)

    minSamlesLabel=Tk.Label(window,text="Minimum samples a ganglion should contain:")
    minSamlesLabel.grid(row=5, sticky="W")
    minSamplesEntry=Tk.Entry(window, width=10)
    minSamplesEntry.grid(row=5, column=1, sticky="E")
    minSamplesEntry.insert(0, 2)

    def extract_and_count():
        # rename variable cause I'm too lazy to refactor code
        filename = czi_path
        filename_base = os.path.splitext(os.path.basename(filename))[0]

        # make out_dir if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)



        czi = aicspylibczi.CziFile(filename)

        if maxIP:
            z_min, z_max = czi.get_dims_shape()[0]['Z']
            z_numbers = range(z_min,z_max)

            def maxIPstack(img_list):
                parsed_list = img_list
                parsed_list = [img if isinstance(img, np.ndarray) else io.imread(img) for img in img_list]
                # now all elements in parsed_list are ndarrays
                maxIP_image = np.maximum.reduce(parsed_list)
                return maxIP_image

            image_list = []
            for z_num in z_numbers:
                image_slice, _ = czi.read_image(C=int(channelSpinBox.get()), Z=z_num)
                image_slice = image_slice[0,0,0,0,0,:,:]
                image_list.append(image_slice)
            img_extracted = maxIPstack(image_list)
        else:
            image_slice, _ = czi.read_image(C=int(channelSpinBox.get()), Z=int(zstackSpinBox.get()))
            image_slice = image_slice[0,0,0,0,0,:,:]
            img_extracted = image_slice


        io.imsave(os.path.join(out_dir,f"{filename_base}_c{int(channelSpinBox.get())}_{'maxIP' if maxIP else f'z{zstackSpinBox.get()}'}.tiff"), img_extracted)

        ### Analyzing
        neurons = io.imread(os.path.join(out_dir,f"{filename_base}_c{int(channelSpinBox.get())}_{'maxIP' if maxIP else f'z{zstackSpinBox.get()}'}.tiff"))
        meta = {"Name": os.path.splitext(f"{filename_base}_c{int(channelSpinBox.get())}_{'maxIP' if maxIP else f'z{zstackSpinBox.get()}'}.tiff")[0]}
        directory = os.path.join(str(out_dir), f"result_{os.path.splitext(meta['Name'])[0]}_{time.strftime('%m'+'_'+'%d'+'_'+'%Y')}")

        if os.path.exists(directory):
            expand = 0
            while True:
                expand += 1
                new_directory = directory+"_"+str(expand)
                if os.path.exists(new_directory):
                    continue
                else:
                    directory = new_directory
                    os.makedirs(directory, exist_ok=True)
                    break
        else:
            os.makedirs(directory, exist_ok=True)

        # Actually process the image and segmetn
        local_maxi, labels, gauss = processing.wide_clusters(neurons,
                                                             sigma=float(sigmaEntry.get()),
                                                             pixel_density=float(pixelDensityEntry.get()),
                                                             min_samples=int(minSamplesEntry.get()),
                                                             meta=meta,
                                                             directory=directory,
                                                             save= True)

        ganglion_prop = processing.segmentation(gauss, local_maxi, labels, meta, directory, save = True)

        # Save the dataframe
        # Run dataframe function from module
        _, _ = analysis.create_dataframe(ganglion_prop, labels, local_maxi, meta, directory, save=True)
        statusLabel.config(text=f"Counting succesful. Output saved to {directory}")
    # Button to close the application:
    def Close():
        window.destroy()

    # Button for closing
    exitButton = Tk.Button(window, text="Exit", activebackground=("red"),command=Close)
    exitButton.grid(row=8, sticky="W")
    statusLabel = Tk.Label(window, text="", fg="green")
    statusLabel.grid(row=9, column=0, sticky="W")


    startBtn = Tk.Button(window, text="Count neurons", bg=("#4ea2a2"), command=extract_and_count)
    startBtn.grid(column=1,row=8, sticky="E")

    window.mainloop()

runApp()
