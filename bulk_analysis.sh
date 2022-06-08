#!/usr/bin/env sh


FILES="./data/confocal_mouse_stomachs/*.czi"
for f in $FILES
do
  echo "Processing $f file..."
  python extract_and_count.py -o ./mouse_stomach_output/ -i -c 0 -p 5 -s 7 -m 2 $f
done
