#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


################################  global variables  ################################
rm -rf ./tests/logs ./tests/output.txt

EXPDIR=./tests/_tests_dl_callbacks
LOGDIR=./tests/logs/_tests_dl_callbacks
CHECKPOINTS=${LOGDIR}/checkpoints
LOGFILE=${CHECKPOINTS}/_metrics.json
EXP_OUTPUT=./tests/output.txt


function check_file_existence {
    # $1 - path to file
    if [[ ! -f "$1" ]]
    then
        echo "There is no '$1'!"
        exit 1
    fi
}


function check_num_files {
    # $1 - ls directory
    # $2 - expected count
    NFILES=$( ls $1 | wc -l )
    if [[ $NFILES -ne $2 ]]
    then
        echo "Different number of files in '$1' - "`
              `"expected $2 but actual number is $NFILES!"
        exit 1
    fi
}


function check_checkpoints {
    # $1 - file prefix
    # $2 - expected count
    check_num_files "${1}.pth" $2
    check_num_files "${1}_full.pth" $2
}


function check_line_counts {
    # $1 file
    # $2 pattern
    # $3 expected count
    ACTUAL_COUNT=$( grep -c "$2" $1 || true )  # '|| true' for handling pipefail
    if [ $ACTUAL_COUNT -ne $3 ]
    then
        echo "Different number of lines in file '$1' - "`
             `"expected $3 (should match '$2') but actual number is $ACTUAL_COUNT!"
        exit 1
    fi
}


################################  pipeline 00 ################################
# setup: ignore accuracy when it is not a main metric

LOG_MSG='pipeline 00'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config21.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "(train)\: accuracy" 5
check_line_counts ${EXP_OUTPUT} "(valid)\: accuracy" 0

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


################################  pipeline 01 ################################
# setup: ignore loss when it is not a main metric

LOG_MSG='pipeline 01'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config22.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "\(train\).* loss" 5
check_line_counts ${EXP_OUTPUT} "\(valid\).* loss" 0

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


################################  pipeline 02 ################################
# setup: different ignore schemes for loaders

LOG_MSG='pipeline 02'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config23.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "(train)\: accuracy" 4
check_line_counts ${EXP_OUTPUT} "(valid)\: accuracy" 4
check_line_counts ${EXP_OUTPUT} "Epoch 2 (valid): loss" 1
check_line_counts ${EXP_OUTPUT} "Epoch 3 (train): loss" 1

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


################################  pipeline 03 ################################
# setup: different ignore schemes for loaders (with duplicated epochs)

LOG_MSG='pipeline 03'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config24.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "(train)\: accuracy" 2
check_line_counts ${EXP_OUTPUT} "(valid)\: accuracy" 3
check_line_counts ${EXP_OUTPUT} "Epoch [134] (train): loss" 3
check_line_counts ${EXP_OUTPUT} "Epoch [25] (valid): loss" 2


check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


################################  pipeline 04 ################################
# setup: eval function from config

LOG_MSG='pipeline 04'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config25.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "(train)\: accuracy" 5
check_line_counts ${EXP_OUTPUT} "(valid)\: accuracy" 6
check_line_counts ${EXP_OUTPUT} "Epoch [2468] (train): loss" 4
check_line_counts ${EXP_OUTPUT} "Epoch [369] (valid): loss" 3


check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


###############################  pipeline 05 ################################
# setup: multiple stages and global epochs

LOG_MSG='pipeline 05'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config26.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "(train)\: accuracy" 4
check_line_counts ${EXP_OUTPUT} "(valid)\: accuracy" 8
check_line_counts ${EXP_OUTPUT} "Epoch [1267] (train): loss" 4


check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 9   # 4x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


###############################  pipeline 06 ################################
# setup: multiple stages with multiple types of epochs
#        and global epochs

LOG_MSG='pipeline 06'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config27.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "(train)\: accuracy" 4
check_line_counts ${EXP_OUTPUT} "(valid)\: accuracy" 4
check_line_counts ${EXP_OUTPUT} "Epoch [2457] (train): loss" 4
check_line_counts ${EXP_OUTPUT} "Epoch [2457] (valid): loss" 4


check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 9   # 4x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


###############################  pipeline 07 ################################
# setup: multiple stages and global epochs and inversed loaders

LOG_MSG='pipeline 07'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config28.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "(train)\: accuracy" 4
check_line_counts ${EXP_OUTPUT} "(valid)\: accuracy" 0
check_line_counts ${EXP_OUTPUT} "Epoch [3458] (train): loss" 4


check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 9   # 4x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


###############################  pipeline 08 ################################
# setup: multiple stages and global epochs and inversed epochs

LOG_MSG='pipeline 08'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config29.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "(train)\: accuracy" 4
check_line_counts ${EXP_OUTPUT} "(valid)\: accuracy" 4
check_line_counts ${EXP_OUTPUT} "Epoch [1368] (train): loss" 4
check_line_counts ${EXP_OUTPUT} "Epoch [1368] (valid): loss" 4


check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 9   # 4x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}