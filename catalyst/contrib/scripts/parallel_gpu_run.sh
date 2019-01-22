#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"

N=${1:-4}
LOGS=${2:-logs/$(date +%s)}

mkdir -p "$LOGS"

PIDS=()
CMD=
IDX=0
echo "$(cat)" | while read cmd ; do
    CMD="$CMD $cmd"
    lastchar="${cmd: -1}"
    if [ '\' == "$lastchar" ] ; then
        continue
    fi
    CMD="${CMD#"${CMD%%[![:space:]]*}"}"
    CMD="${CMD%"${CMD##*[![:space:]]}"}"
    if [ -z "$CMD" ] ; then
        continue
    fi
    started=
    while [ -z "$started" ] ; do
        for ((gpu=0; gpu<N; gpu++)) ; do
            pid=${PIDS[$gpu]}
            if [ -n "$pid" ] && ps -p $pid > /dev/null; then
                continue
            fi
            echo "RUN [GPU $gpu]: $CMD" | tee "$LOGS/$IDX.log"
            CUDA_VISIBLE_DEVICES=$gpu bash -c "$CMD" >>"$LOGS/$IDX.log" 2>>"$LOGS/$IDX.err" || \
                echo $IDX >> "$LOGS/$IDX.die" &
            pid=$!
            echo $pid >> "$LOGS/pids.txt"
            PIDS[$gpu]=$pid
            started=1
            IDX=$((IDX+1))
            break
        done
        if [ -z "$started" ] ; then
            sleep 0.1
        fi
    done
    CMD=""
done

for pid in $(cat "$LOGS/pids.txt") ; do
    while ps -p $pid > /dev/null ; do
        sleep 0.1
    done
done
rm "$LOGS/pids.txt"