#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v


function check_file_existence {
    if [[ ! -f $1 ]]
    then
        echo "There is no '$1'!";
        exit 1
    fi
}


################################  preparing  ################################
rm -rf ./tests/logs

EXPDIR=./tests/_tests_dl_callbacks
LOGDIR=./tests/logs/_tests_dl_callbacks
LOGFILE=${LOGDIR}/checkpoints/_metrics.json


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

cat $LOGFILE;
echo ${LOG_MSG};

check_file_existence "${LOGDIR}/checkpoints/_metrics.json"

for FPREFIX in 'best' 'last' 'stage1.5'
do
    check_file_existence "${LOGDIR}/checkpoints/${FPREFIX}.pth"
    check_file_existence "${LOGDIR}/checkpoints/${FPREFIX}_full.pth"
done

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

check_file_existence "${LOGDIR}/checkpoints/_metrics.json"

for FPREFIX in 'best' 'last' 'stage1.5'
do
    check_file_existence "${LOGDIR}/checkpoints/${FPREFIX}.pth"
    check_file_existence "${LOGDIR}/checkpoints/${FPREFIX}_full.pth"
done

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

cat $LOGFILE;
echo ${LOG_MSG};

for FPREFIX in 'best' 'last' 'stage1.5' 'stage2.4' 'stage3.5'
do
    check_file_existence "${LOGDIR}/checkpoints/${FPREFIX}.pth"
    check_file_existence "${LOGDIR}/checkpoints/${FPREFIX}_full.pth"
done

rm -rf ${LOGDIR}


################################  pipeline 03  ################################
# checking with three checkpoint and one stage
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

check_file_existence "${LOGDIR}/checkpoints/_metrics.json"

for FPREFIX in 'best' 'last' 'stage1.3' 'stage1.4' 'stage1.5'
do
    check_file_existence "${LOGDIR}/checkpoints/${FPREFIX}.pth"
    check_file_existence "${LOGDIR}/checkpoints/${FPREFIX}_full.pth"
done

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

cat $LOGFILE;
echo ${LOG_MSG};

check_file_existence "${LOGDIR}/checkpoints/_metrics.json"

for FPREFIX in 'best' 'last' 'stage1.3' 'stage1.4' 'stage1.5' 'stage2.3' 'stage2.4' 'stage2.5' 'stage3.3' 'stage3.4' 'stage3.5'
do
    check_file_existence "${LOGDIR}/checkpoints/${FPREFIX}.pth"
    check_file_existence "${LOGDIR}/checkpoints/${FPREFIX}_full.pth"
done

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

cat $LOGFILE;
echo ${LOG_MSG};

check_file_existence "${LOGDIR}/checkpoints/_metrics.json"

for FPREFIX in 'best' 'last'
do
    check_file_existence "${LOGDIR}/checkpoints/${FPREFIX}.pth"
    check_file_existence "${LOGDIR}/checkpoints/${FPREFIX}_full.pth"
done

rm -rf ${LOGDIR}


################################  pipeline 06  ################################
# checking with three checkpoint and one stage
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

cat $LOGFILE;
echo ${LOG_MSG};

check_file_existence "${LOGDIR}/checkpoints/_metrics.json"

for FPREFIX in 'best' 'last'
do
    check_file_existence "${LOGDIR}/checkpoints/${FPREFIX}.pth"
    check_file_existence "${LOGDIR}/checkpoints/${FPREFIX}_full.pth"
done

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

cat $LOGFILE;
echo ${LOG_MSG}

check_file_existence "${LOGDIR}/checkpoints/_metrics.json"

for FPREFIX in 'best' 'last' 'stage1.5'
do
    check_file_existence "${LOGDIR}/checkpoints/${FPREFIX}.pth"
    check_file_existence "${LOGDIR}/checkpoints/${FPREFIX}_full.pth"
done

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

cat $LOGFILE;
echo ${LOG_MSG};

check_file_existence "${LOGDIR}/checkpoints/_metrics.json"

for FPREFIX in 'best' 'last' 'stage1.3' 'stage1.4' 'stage1.5' 'stage2.3' 'stage2.4' 'stage2.5' 'stage3.3' 'stage3.4' 'stage3.5'
do
    check_file_existence "${LOGDIR}/checkpoints/${FPREFIX}.pth"
    check_file_existence "${LOGDIR}/checkpoints/${FPREFIX}_full.pth"
done

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

cat $LOGFILE;
echo ${LOG_MSG};

check_file_existence "${LOGDIR}/checkpoints/_metrics.json"

for FPREFIX in 'best' 'last' 'stage1.5' 'stage2.5'
do
    check_file_existence "${LOGDIR}/checkpoints/${FPREFIX}.pth"
    check_file_existence "${LOGDIR}/checkpoints/${FPREFIX}_full.pth"
done

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

cat $LOGFILE;
echo ${LOG_MSG};

check_file_existence "${LOGDIR}/checkpoints/_metrics.json"

for FPREFIX in 'best' 'last' 'stage1.3' 'stage1.4' 'stage1.5' 'stage2.3' 'stage2.4' 'stage2.5'
do
    check_file_existence "${LOGDIR}/checkpoints/${FPREFIX}.pth"
    check_file_existence "${LOGDIR}/checkpoints/${FPREFIX}_full.pth"
done

rm -rf ${LOGDIR}
