#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


################################  global variables  ################################
rm -rf ./tests/logs

EXPDIR=./tests/_tests_dl_callbacks
LOGDIR=./tests/logs/_tests_dl_callbacks
CHECKPOINTS=${LOGDIR}/checkpoints
LOGFILE=${CHECKPOINTS}/_metrics.json


function check_file_existence {
    if [[ ! -f $1 ]]
    then
        echo "There is no '$1'!"
        exit 1
    fi
}


function check_num_files {
    NFILES=$( ls $1 | wc -l )
    if [[ $NFILES -ne $2 ]]
    then
        echo "Different number of files in '$1' - expected $2 but actual number is $NFILES!"
        exit 1
    fi
}


function check_checkpoints {
    check_num_files "${CHECKPOINTS}/${1}.pth" $2
    check_num_files "${CHECKPOINTS}/${1}_full.pth" $2
}


################################  pipeline 00  ################################
# checking dafult parameters of checkpoint and one stage
LOG_MSG='pipeline 00'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config0.yml \
  --logdir=${LOGDIR}

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat ${LOGFILE}
echo ${LOG_MSG}

check_file_existence ${LOGFILE}
check_checkpoints "best" 1
check_checkpoints "last" 1
check_checkpoints "stage1\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7

rm -rf ${LOGDIR}


################################  pipeline 01  ################################
# checking with one checkpoint and one stage
LOG_MSG='pipeline 01'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config1.yml \
  --logdir=${LOGDIR}

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat $LOGFILE;
echo ${LOG_MSG};

check_file_existence ${LOGFILE}
check_checkpoints "best" 1
check_checkpoints "last" 1
check_checkpoints "stage1\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7  # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR}


################################  pipeline 02  ################################
# checking with one checkpoint and three stages
LOG_MSG='pipeline 02'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config2.yml \
  --logdir=${LOGDIR}

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat ${LOGFILE}
echo ${LOG_MSG}

check_file_existence ${LOGFILE}
check_checkpoints "best" 1
check_checkpoints "last" 1
check_checkpoints "stage1\.[[:digit:]]" 1
check_checkpoints "stage2\.[[:digit:]]" 1
check_checkpoints "stage3\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 11  # 5x2 checkpoints + metrics.json

rm -rf ${LOGDIR}


################################  pipeline 03  ################################
# checking with three checkpoints and one stage
LOG_MSG='pipeline 03'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config3.yml \
  --logdir=${LOGDIR}

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat $LOGFILE;
echo ${LOG_MSG};

check_file_existence ${LOGFILE}
check_checkpoints "best" 1
check_checkpoints "last" 1
check_checkpoints "stage1\.[[:digit:]]" 3
check_num_files ${CHECKPOINTS} 11  # 5x2 checkpoints + metrics.json

rm -rf ${LOGDIR}


################################  pipeline 04  ################################
# checking with three checkpoint and three stages
LOG_MSG='pipeline 04'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config4.yml \
  --logdir=${LOGDIR}

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat ${LOGFILE}
echo ${LOG_MSG}

check_file_existence ${LOGFILE}
check_checkpoints "best" 1
check_checkpoints "last" 1
check_checkpoints "stage1\.[[:digit:]]" 3
check_checkpoints "stage2\.[[:digit:]]" 3
check_checkpoints "stage3\.[[:digit:]]" 3
check_num_files ${CHECKPOINTS} 23  # 11x2 checkpoints + metrics.json

rm -rf ${LOGDIR}


################################  pipeline 05  ################################
# checking with zero checkpoints and one stage
LOG_MSG='pipeline 05'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config5.yml \
  --logdir=${LOGDIR}

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat ${LOGFILE}
echo ${LOG_MSG}

check_file_existence ${LOGFILE}
check_checkpoints "best" 1
check_checkpoints "last" 1
check_num_files ${CHECKPOINTS} 5  # 2x2 checkpoints + metrics.json

rm -rf ${LOGDIR}


################################  pipeline 06  ################################
# checking with zepo checkpoints and one stage
LOG_MSG='pipeline 06'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config6.yml \
  --logdir=${LOGDIR}

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat ${LOGFILE}
echo ${LOG_MSG}

check_file_existence ${LOGFILE}
check_checkpoints "best" 1
check_checkpoints "last" 1
check_num_files ${CHECKPOINTS} 5  # 2x2 checkpoints + metrics.json

rm -rf ${LOGDIR}


################################  pipeline 07  ################################
# checking with one checkpoint and one stage
LOG_MSG='pipeline 07'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config7.yml \
  --logdir=${LOGDIR}

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat ${LOGFILE}
echo ${LOG_MSG}

check_file_existence ${LOGFILE}
check_checkpoints "best" 1
check_checkpoints "last" 1
check_checkpoints "stage1\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7  # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR}


################################  pipeline 08  ################################
# checking with three checkpoints and three stages
LOG_MSG='pipeline 08'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config8.yml \
  --logdir=${LOGDIR}

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat ${LOGFILE}
echo ${LOG_MSG}

check_file_existence ${LOGFILE}
check_checkpoints "best" 1
check_checkpoints "last" 1
check_checkpoints "stage1\.[[:digit:]]" 3
check_checkpoints "stage2\.[[:digit:]]" 3
check_checkpoints "stage3\.[[:digit:]]" 3
check_num_files ${CHECKPOINTS} 23  # 11x2 checkpoints + metrics.json

rm -rf ${LOGDIR}


################################  pipeline 09  ################################
# checking with one checkpoint and two stages 
# with different ''load_on_stage_end'' options
LOG_MSG='pipeline 09'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config9.yml \
  --logdir=${LOGDIR}

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat ${LOGFILE}
echo ${LOG_MSG}

check_file_existence ${LOGFILE}
check_checkpoints "best" 1
check_checkpoints "last" 1
check_checkpoints "stage1\.[[:digit:]]" 1
check_checkpoints "stage2\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 9  # 4x2 checkpoints + metrics.json

rm -rf ${LOGDIR}


################################  pipeline 10  ################################
# checking with three checkpoints and two stages 
# with different ''load_on_stage_end'' options
LOG_MSG='pipeline 10'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config10.yml \
  --logdir=${LOGDIR}

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat ${LOGFILE}
echo ${LOG_MSG}

check_file_existence ${LOGFILE}
check_checkpoints "best" 1
check_checkpoints "last" 1
check_checkpoints "stage1\.[[:digit:]]" 3
check_checkpoints "stage2\.[[:digit:]]" 3
check_num_files ${CHECKPOINTS} 17  # 8x2 checkpoints + metrics.json

rm -rf ${LOGDIR}


################################  pipeline 11  ################################
# checking with three stages and default not specified callback
# (CheckpointCallback is one of default callbacks)
LOG_MSG='pipeline 11'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config11.yml \
  --logdir=${LOGDIR}

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat ${LOGFILE}
echo ${LOG_MSG}

check_file_existence ${LOGFILE}
check_checkpoints "best" 1
check_checkpoints "last" 1
check_checkpoints "stage1\.[[:digit:]]" 1
check_checkpoints "stage2\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 9  # 4x2 checkpoints + metrics.json

rm -rf ${LOGDIR}
