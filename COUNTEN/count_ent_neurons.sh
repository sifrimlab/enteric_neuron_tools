#!/bin/bash

for file in /media/Puzzles/enteric_neurones/ME_distal_18_weeks/extracted/*.tiff
do
        python count_ent_neurons.py "$file" /media/Puzzles/enteric_neurones/ME_distal_18_weeks/extracted/output/
done
