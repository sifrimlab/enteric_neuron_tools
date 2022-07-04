# Enteric Neuron Counting

This repository is contains code that can be used to process .czi images that contain fluorescent images of enteric neurons. It's mostly a wrapper around the enteric neuron counter created by https://github.com/KLab-JHU/COUNTEN, so all credit goes to them. What this repository adds is the fact that it reads and extract the necessary images and maximum intensity projection from a raw .czi image and saves the output in a meaningfull way. This makes it so that it can be run from command line in a script to increase throughput immensely.

In addition to that, the necessary tools are also there to use pyinstaller and create an executable (.exe) that can be run from Windows to be user friendly to anyone, regardless of command-line familiarity.


## Dependencies
- numpy
- pandas
- scikit-image
- scikit-learn
- aicspylibczi
- tk (only for the GUI application)
## Usage

### Installation
Clone this repo locally:
```bash
git clone https://github.com/sifrimlab/enteric_neuron_tools
cd enteric_neuron_tools
```

Install dependencies using pip, or use the ent_neurons.yml file to create a conda environment:

```bash
conda env create --file=ent_neurons.yml --prefix ./ent_neurons_env/
conda activate ./ent_neurons_env/
```

### Running the program

Either run the python script with the appropriate arguments, which can be found by running:
```bash
python extract_and_count.py -h
```

Or run the simple GUI application: 
```bash
python ui_extract_and_count.py 
```

## Building a Windows executable (.exe)
1) This needs to be done from a windows environment, so install conda (miniconda) in a Windows environment and activate a miniconda cmd prompt
2) Clone this repo as described above.
2) Create a conda environment and activate is as described above.
3) Additionally install pyinstaller using:
```bash
pip install pyinstaller
```
4) Create the executable using the following command:
```bash
pyinstaller --onefile --noconsole --add-data .\imgs;. --additional-hooks-dir=%cd% -n count_neurons ui_extract_and_count.py
```


