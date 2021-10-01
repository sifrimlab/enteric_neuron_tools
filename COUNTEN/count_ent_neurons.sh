#!/bin/bash

for file in /media/tool/enteric_neurones/ME_distal_18_weeks/extracted/*
do
        python count_ent_neurons.py "$file" /media/tool/enteric_neurones/ME_distal_18_weeks/extracted/output/
done
