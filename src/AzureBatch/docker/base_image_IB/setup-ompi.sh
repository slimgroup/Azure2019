#!/usr/bin/env bash

# get nodes and compute number of processors
IFS=',' read -ra HOSTS <<< "$AZ_BATCH_HOST_LIST"
nodes=${#HOSTS[@]}
ppn=$nsockets
np=$(($nodes * $ppn))


# create hostfile
hostfile="hostfile"
touch $hostfile
>| $hostfile
for node in "${HOSTS[@]}"
do
    echo $node slots=$ppn max-slots=$ppn >> $hostfile
done

mpienvopts=`echo \`env | grep "HPCX_" | sed -e "s/=.*$//"\` | sed -e "s/ / -x /g"`
devitoenvopts=`echo \`env | grep "DEVITO_" | sed -e "s/=.*$//"\` | sed -e "s/ / -x /g"`
ompenvopts=`echo \`env | grep "OMP_" | sed -e "s/=.*$//"\` | sed -e "s/ / -x /g"`
mpienvopts="$mpienvopts -x PATH -x LD_LIBRARY_PATH"
export mpienvopts
export devitoenvopts
export ompenvopts
export np
export hostfile
