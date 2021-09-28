#!/bin/bash

for file in /media/tool/enteric_neurones/ME_proximal_18_weeks_march/*.czi
do
        python extract_neurons.py "$file"
done
