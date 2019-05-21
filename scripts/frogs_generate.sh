#!/bin/bash

git clone https://github.com/jonshamir/frog-dataset.git ../frogs_temp/
cp -R ../frogs_temp/data-64 ../../MLApplicationsForGraphics
mv ../data-64 ../frogs-64
rm -R -f ../frogs_temp
echo "done generating"