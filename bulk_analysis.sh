#!/usr/bin/env sh


FILES="./mouse_stomach_output/ileum/redo/*.czi"
mkdir ./mouse_stomach_output/ileum/redo/redone/
for f in $FILES
do
  echo "Processing $f file..."
  python extract_and_count.py -o ./mouse_stomach_output/ileum/redo/redone/ -i -c 0 -p 5 -s 15 -m 2 $f
done
