#!/bin/bash  

for((i=0;i<3000;i+=200));
do
python3 download.py $i > ${i}.log 2>&1 &                                                                                                                
done
