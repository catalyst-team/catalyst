#!/usr/bin/env bash

# usage:
# ./ssh_port_forward host local_ip port_start port_end

cmd="ssh $1 -f -N -R localhost:9090:$2:9090"
for i in `seq $3 $4`; do
  cmd="$cmd -L $i:localhost:$i"
done

$cmd