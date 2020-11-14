#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


################################  global variables  ################################
rm -rf ./tests/logs ./tests/output.txt

EXPDIR=./tests/_tests_contrib_dl_callbacks
LOGDIR=./tests/logs/_tests_contrib_dl_callbacks
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


################################  pipeline 00  ################################
# setup: run validation once in 2 epoch
LOG_MSG='pipeline 00'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config0.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "(train):" 5
check_line_counts ${EXP_OUTPUT} "(valid):" 2
check_line_counts ${EXP_OUTPUT} ".*/stage1\.[[:digit:]]\.pth" 1

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_num_files "${CHECKPOINTS}/stage1.*.pth" 2
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


################################  pipeline 01  ################################
# setup: never run validation will raise an error
LOG_MSG='pipeline 01'
echo ${LOG_MSG}

{
  # try
  PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config1.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}
} || {  # catch
    echo "Successfully raised error caused by validation period 0!"
}

cat ${EXP_OUTPUT}

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

rm -rf ${LOGDIR} ${EXP_OUTPUT}


################################  pipeline 02  ################################
# setup: multiple loaders
LOG_MSG='pipeline 02'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config2.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "(train):" 10
check_line_counts ${EXP_OUTPUT} "(train_additional):" 5
check_line_counts ${EXP_OUTPUT} "(valid):" 3
check_line_counts ${EXP_OUTPUT} "(valid_additional):" 0
check_line_counts ${EXP_OUTPUT} ".*/stage1\.[[:digit:]]\{1,2\}\.pth" 1

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_num_files "${CHECKPOINTS}/stage1.*.pth" 2
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


################################  pipeline 03  ################################
# setup: multiple loaders and few stages
LOG_MSG='pipeline 03'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config3.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "(train):" 15
check_line_counts ${EXP_OUTPUT} "(train_additional):" 5
check_line_counts ${EXP_OUTPUT} "(valid):" 6
check_line_counts ${EXP_OUTPUT} "(valid_additional):" 0
check_line_counts ${EXP_OUTPUT} ".*/stage1\.[[:digit:]]\.pth" 1
check_line_counts ${EXP_OUTPUT} ".*/stage2\.[[:digit:]]\{1,2\}\.pth" 1

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_num_files "${CHECKPOINTS}/stage1.*.pth" 2
check_num_files "${CHECKPOINTS}/stage2.*.pth" 2
check_num_files ${CHECKPOINTS} 9   # 4x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


################################  pipeline 04  ################################
# setup: test for raising error when there is no loader in epoch

LOG_MSG='pipeline 04'
echo ${LOG_MSG}

{
  # try
  PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config4.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}
} || {  # catch
    echo "Successfully raised error for epoch with no loaders!"
}

cat ${EXP_OUTPUT}

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

rm -rf ${LOGDIR} ${EXP_OUTPUT}


################################  pipeline 05  ################################
# setup: test for ignoring some random loaders

LOG_MSG='pipeline 05'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config5.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "(train):" 10
check_line_counts ${EXP_OUTPUT} "(train_additional):" 5
check_line_counts ${EXP_OUTPUT} "(train_not_exists):" 0
check_line_counts ${EXP_OUTPUT} "(valid):" 3
check_line_counts ${EXP_OUTPUT} "(valid_additional):" 0
check_line_counts ${EXP_OUTPUT} "(valid_not_exist):" 0
check_line_counts ${EXP_OUTPUT} ".*/stage1\.[[:digit:]]\{1,2\}\.pth" 1

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_num_files "${CHECKPOINTS}/stage1.*.pth" 2
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}
