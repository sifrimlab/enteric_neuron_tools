import sys
import os
import argparse
import custom_io
import processing
import analysis
import tkinter as Tk
from tkinter.filedialog import askopenfilename, askdirectory
import importlib.util # This is to manually import the scripts to make sure it's all correct
import matplotlib.pyplot as plt
import numpy as np
import aicspylibczi
from skimage import io
import javabridge
import bioformats

# First extract the correct image from the czi stack

def runApp():
    window = Tk.Tk()
    window.title("Count enteric neurons")
    window.geometry('900x200')

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
    fileLabel.grid(row=0, column=1)

    # output directory
    outputDirButton = Tk.Button(window,fg="red", text="Select your output directory: ", command=openDirectory)
    outputDirLabel=Tk.Label(window, text="No directory selected")
    outputDirButton.grid(row=1, sticky="W")
    outputDirLabel.grid(row=1, column=1)


    # Fill in the channel number
    channelLabel=Tk.Label(window,text="Channel index of marker of interest:")
    channelLabel.grid(row=2, sticky="W")
    channelSpinBox = Tk.Spinbox(from_=0,to=4)
    # channelSpinBox.pack()
    channelSpinBox.grid(row=2, column=1)

    # Entries for parameters
    pixelDensityLabel=Tk.Label(window,text="Pixel density (in pixel/micrometer):")
    pixelDensityLabel.grid(row=3, sticky="W")
    pixelDensityEntry=Tk.Entry(window, width=10)
    pixelDensityEntry.grid(row=3, column=1)
    pixelDensityEntry.insert(0, 3.2)

    sigmaLabel=Tk.Label(window,text="Sigma parameter that controls background smoothing:")
    sigmaLabel.grid(row=4, sticky="W")
    sigmaEntry=Tk.Entry(window, width=10)
    sigmaEntry.grid(row=4, column=1)
    sigmaEntry.insert(0, 7)

    minSamlesLabel=Tk.Label(window,text="Minimum samples a ganglion should contain:")
    minSamlesLabel.grid(row=5, sticky="W")
    minSamplesEntry=Tk.Entry(window, width=10)
    minSamplesEntry.grid(row=5, column=1)
    minSamplesEntry.insert(0, 2)

    def extract_and_count():
        # rename variable cause I'm too lazy to refactor code
        filename = czi_path
        filename_base = os.path.splitext(os.path.basename(filename))[0]

        # make out_dir if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)



        czi = aicspylibczi.CziFile(filename)

        z_min, z_max = czi.get_dims_shape()[0]['Z']
        z_numbers = range(z_min,z_max)

        def maxIPstack(img_list):
            parsed_list = img_list
            parsed_list = [img if isinstance(img, np.ndarray) else io.imread(img) for img in img_list]
            # now all elements in parsed_list are ndarrays
            maxIP = np.maximum.reduce(parsed_list)
            return maxIP

        image_list = []
        for z_num in z_numbers:
            image_slice, _ = czi.read_image(C=int(channelSpinBox.get()), Z=z_num)
            image_slice = image_slice[0,0,0,0,0,:,:]
            image_list.append(image_slice)
        img_extracted = maxIPstack(image_list)

        io.imsave(os.path.join(out_dir,f"{filename_base}_c{int(channelSpinBox.get())}_maxIP.tiff"), img_extracted)

        ### Analyzing

        script_path = os.path.join(os.path.dirname(sys.argv[0]),"scripts/")


        # Load module
        # spec = importlib.util.spec_from_file_location("io",os.path.join(script_path, "io.py"))
        # custom_io = importlib.util.module_from_spec(spec)
        # spec.loader.exec_module(custom_io)

        javabridge.start_vm(class_path=bioformats.JARS)

        neurons, directory, meta = custom_io.load_TIFF(os.path.join(out_dir,f"{filename_base}_c{int(channelSpinBox.get())}_maxIP.tiff"), out_dir)

        # javabridge.kill_vm()

        # Load module
        # spec = importlib.util.spec_from_file_location("processing",os.path.join(script_path,"processing.py"))
        # processing = importlib.util.module_from_spec(spec)
        # spec.loader.exec_module(processing)

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

        # Load module
        # spec = importlib.util.spec_from_file_location("analysis",os.path.join(script_path,"analysis.py"))
        # analysis = importlib.util.module_from_spec(spec)
        # spec.loader.exec_module(analysis)

        # Run dataframe function from module
        _, _ = analysis.create_dataframe(ganglion_prop, labels, local_maxi, meta, directory, save=True)
    # Button to close the application:
    def Close():
        try:
            javabridge.kill_vm()
        except:
            pass
        window.destroy()
  
  
    # Button for closing
    exitButton = Tk.Button(window, text="Exit", activebackground=("red"),command=Close)
    exitButton.grid(row=6, sticky="W")


    startBtn = Tk.Button(window, text="Count neurons", bg=("#4ea2a2"), command=extract_and_count)
    startBtn.grid(column=1,row=6)

    window.mainloop()

runApp()
