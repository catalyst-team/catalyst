#!/usr/bin/env bash

#usage:
# CUDA_VISIBLE_DEVICES=0 ./parallel_run.sh \
#    --redis-port 12001 \
#    --config ..../config.yml \
#    --logdir ..../logdir/log-prefix \
#    --param-name "trainer/start_learning" \
#    --param-values "10, 20, 30" \
#    --param-type "int" \
#    --wait-time 30 \
#    --n-trials 3

REDIS_PORT=
CONFIG=
LOGDIR=
PARAM_NAME=
PARAM_VALUES=
PARAM_TYPE=
WAIT_TIME=
N_TRIALS=
#GPUS=
#CHECK=0


while (( "$#" )); do
  case "$1" in
    --redis-port)
      REDIS_PORT=$2
      shift 2
      ;;
    --config)
      CONFIG=$2
      shift 2
      ;;
    --logdir)
      LOGDIR=$2
      shift 2
      ;;
    --param-name)
      PARAM_NAME=$2
      shift 2
      ;;
    --param-values)
      PARAM_VALUES=$2
      shift 2
      ;;
    --param-type)
      PARAM_TYPE=$2
      shift 2
      ;;
    --wait-time)
      WAIT_TIME=$2
      shift 2
      ;;
    --n-trials)
      N_TRIALS=$2
      shift 2
      ;;
#    --gpus)
#      GPUS=$2
#      shift 2
#      ;;
#    --check)
#      CHECK=1
#      shift 1
#      ;;
    *) # preserve positional arguments
      shift
      ;;
  esac
done


IFS=',' read -r -a PARAM_VALUES <<< "$PARAM_VALUES"
#IFS=',' read -r -a GPUS <<< "GPUS"
#GPUS_LEN="${#GPUS[@]}"

for param_value in  ${PARAM_VALUES[@]}
do
    param_name=${PARAM_NAME/"/"/"-"}
    for ((trial=0; trial<N_TRIALS; trial++)) ; do
        redis-server --port "${REDIS_PORT}" &
        pid1=$!
        sleep 10
        catalyst-rl run-trainer \
          --config="${CONFIG}" \
          --logdir="${LOGDIR}-${param_name}-${param_value}-${trial}" \
          "--${PARAM_NAME}=${param_value}:${PARAM_TYPE}" \
          --redis/port="${REDIS_PORT}:int" &
        pid2=$!
        sleep 20
        CUDA_VISIBLE_DEVICES="" catalyst-rl run-samplers \
          --config="${CONFIG}" \
          --logdir="${LOGDIR}-${param_name}-${param_value}-${trial}" \
          "--${PARAM_NAME}=${param_value}:${PARAM_TYPE}" \
          --redis/port="${REDIS_PORT}:int" &
        pid3=$!
        sleep 30
        sleep ${WAIT_TIME}
        kill -9 $pid3
        sleep 10
        kill -9 $pid2
        sleep 10
        kill -9 $pid1
        sleep 60
    done
done


#LOGS=${2:-logs/$(date +%s)}
#mkdir -p "$LOGS"
#
#PIDS=()
#CMD=
#IDX=0
#started=
#while [ -z "$started" ] ; do
#    for ((gpu_num=0; gpu_num<GPUS_LEN; gpu_num++)) ; do
#        pid=${PIDS[$gpu_num]}
#        if [ -n "$pid" ] && ps -p $pid > /dev/null; then
#            continue
#        fi
#        gpu=${GPUS[$gpu_num]}
#        echo "RUN [GPU $gpu]: $CMD" | tee "$LOGS/$IDX.log"
#
#
#        CUDA_VISIBLE_DEVICES=$gpu bash -c "$CMD" >>"$LOGS/$IDX.log" 2>>"$LOGS/$IDX.err" || \
#            echo $IDX >> "$LOGS/$IDX.die" &
#        pid=$!
#        echo $pid >> "$LOGS/pids.txt"
#        PIDS[$gpu_num]=$pid
#        started=1
#        IDX=$((IDX+1))
#        break
#    done
#    if [ -z "$started" ] ; then
#        sleep 0.1
#    fi
#done
#
#for pid in $(cat "$LOGS/pids.txt") ; do
#    while ps -p $pid > /dev/null ; do
#        sleep 0.1
#    done
#done
#rm "$LOGS/pids.txt"
