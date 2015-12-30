#!/bin/bash
FILES=*.cu

for f in $FILES
do
  echo "Compiling $f to ${f%.*}.exe ..."
  nvcc $f -o  ${f%.*}.exe
done
