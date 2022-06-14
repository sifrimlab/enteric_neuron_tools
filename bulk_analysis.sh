#!/usr/bin/env sh


FILES="./data/confocal_mouse_stomachs/redo/*.czi"
for f in $FILES
do
  echo "Processing $f file..."
  python extract_and_count.py -o ./mouse_stomach_output/redo/ -i -c 0 -p 5 -s 17 -m 2 $f
done
