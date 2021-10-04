#!/bin/bash

for file in /media/Puzzles/enteric_neurones/ME_prox_18_weeks/*.czi
do
        python extract_neurons.py "$file"
done
