#!/bin/bash
'''
FOR TRANSFERRING FILES                    
NOTE: WE ARE ONLY USING z=0 (snapshot 90) 
'''
source_ep="SOURCE ENDPOINT"
dest_ep="DESTINATION ENDPOINT"

source_dir="/FOF_Subfind/IllustrisTNG/LH/"
dest_dir=""

rm bf.txt
touch bf.txt

for num in {0..1000}; do
echo "LH_$num/groups_090.hdf5 data_$num.hdf5" >> bf.txt
done

mkdir -p LH_$num
globus transfer $source_ep:$source_dir $dest_ep:$dest_dir --batch - < bf.txt
