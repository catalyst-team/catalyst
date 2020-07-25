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

################################  pipeline 00  ################################
# checking default parameters of checkpoint and one stage
LOG_MSG='pipeline 00'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config0.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 0

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


################################  pipeline 01  ################################
# checking with one checkpoint and one stage
LOG_MSG='pipeline 01'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config1.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 0

check_file_existence ${LOGFILE}
cat $LOGFILE;
echo ${LOG_MSG};

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7  # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


################################  pipeline 02  ################################
# checking with one checkpoint and three stages
LOG_MSG='pipeline 02'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config2.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 2
check_line_counts ${EXP_OUTPUT} "=> Loading .*best\.pth" 2

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 1
check_checkpoints "${CHECKPOINTS}/stage3\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 11  # 5x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


################################  pipeline 03  ################################
# checking with three checkpoints and one stage
LOG_MSG='pipeline 03'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config3.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 0

check_file_existence ${LOGFILE}
cat $LOGFILE;
echo ${LOG_MSG};

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 3
check_num_files ${CHECKPOINTS} 11  # 5x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


################################  pipeline 04  ################################
# checking with three checkpoint and three stages
LOG_MSG='pipeline 04'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config4.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 2
check_line_counts ${EXP_OUTPUT} "=> Loading .*best\.pth" 2

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 3
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 3
check_checkpoints "${CHECKPOINTS}/stage3\.[[:digit:]]" 3
check_num_files ${CHECKPOINTS} 23  # 11x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


################################  pipeline 05  ################################
# checking with zero checkpoints and one stage
LOG_MSG='pipeline 05'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config5.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 0

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_num_files ${CHECKPOINTS} 5  # 2x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


################################  pipeline 06  ################################
# checking with zepo checkpoints and three stages
# spoiler - no loading at the end of a stage, only load last state at stage start
# this means that will be default behaviour - loaded best state
LOG_MSG='pipeline 06'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config6.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 2
check_line_counts ${EXP_OUTPUT} "=> Loading .*best\.pth" 2

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_num_files ${CHECKPOINTS} 5  # 2x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


################################  pipeline 07  ################################
# checking with one checkpoint and one stage
LOG_MSG='pipeline 07'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config7.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 0

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7  # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


################################  pipeline 08  ################################
# checking with three checkpoints and three stages
LOG_MSG='pipeline 08'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config8.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 5
check_line_counts ${EXP_OUTPUT} "=> Loading .*best\.pth" 2
check_line_counts ${EXP_OUTPUT} "=> Loading .*best_full\.pth" 3

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 3
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 3
check_checkpoints "${CHECKPOINTS}/stage3\.[[:digit:]]" 3
check_num_files ${CHECKPOINTS} 23  # 11x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


################################  pipeline 09  ################################
# checking with one checkpoint and two stages
# with different ''load_on_stage_end'' options
LOG_MSG='pipeline 09'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config9.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 2
check_line_counts ${EXP_OUTPUT} "=> Loading .*best\.pth" 2

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 9  # 4x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


################################  pipeline 10  ################################
# checking with three checkpoints and two stages
# with different ''load_on_stage_end'' options
LOG_MSG='pipeline 10'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config10.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 2
check_line_counts ${EXP_OUTPUT} "=> Loading .*best\.pth" 2

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 3
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 3
check_num_files ${CHECKPOINTS} 17  # 8x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


################################  pipeline 11  ################################
# checking with three stages and default not specified callback
# (CheckpointCallback is one of default callbacks)
LOG_MSG='pipeline 11'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config11.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 1
check_line_counts ${EXP_OUTPUT} "=> Loading .*best\.pth" 1

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 9  # 4x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}

################################  pipeline 12  ################################
# testing resume option
LOG_MSG='pipeline 12'
echo ${LOG_MSG}

LOGDIR=./tests/logs/_tests_dl_callbacks/for_resume
CHECKPOINTS=${LOGDIR}/checkpoints
LOGFILE=${CHECKPOINTS}/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config0.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 0

check_file_existence $LOGFILE
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_num_files $CHECKPOINTS 7   # 3x2 checkpoints + metrics.json

LOGDIR=./tests/logs/_tests_dl_callbacks/resumed
CHECKPOINTS=${LOGDIR}/checkpoints
LOGFILE=${CHECKPOINTS}/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config12.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 1

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

rm -rf ./tests/logs/_tests_dl_callbacks ${EXP_OUTPUT}


################################  pipeline 13  ################################
# testing resume and resume_dir option
LOG_MSG='pipeline 13'
echo ${LOG_MSG}

LOGDIR=./tests/logs/_tests_dl_callbacks/for_resume
CHECKPOINTS=${LOGDIR}/checkpoints
LOGFILE=${CHECKPOINTS}/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config0.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 0

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

LOGDIR=./tests/logs/_tests_dl_callbacks/resumed
CHECKPOINTS=${LOGDIR}/checkpoints
LOGFILE=${CHECKPOINTS}/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config13.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 1

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

rm -rf ./tests/logs/_tests_dl_callbacks ${EXP_OUTPUT}

################################  pipeline 14  ################################
# testing ''on_stage_start'' option
LOG_MSG='pipeline 14'
echo ${LOG_MSG}

LOGDIR=./tests/logs/_tests_dl_callbacks
CHECKPOINTS=${LOGDIR}/checkpoints
LOGFILE=${CHECKPOINTS}/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config14.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 2
check_line_counts ${EXP_OUTPUT} "=> Loading .*last\.pth" 1
check_line_counts ${EXP_OUTPUT} "=> Loading .*best\.pth" 1

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 2
check_checkpoints "${CHECKPOINTS}/stage3\.[[:digit:]]" 3
check_num_files ${CHECKPOINTS} 17   # 8x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}

################################  pipeline 15  ################################
# testing on_stage_start option with different loading states
LOG_MSG='pipeline 15'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config15.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 4
check_line_counts ${EXP_OUTPUT} "=> Loading .*best_full\.pth" 2
check_line_counts ${EXP_OUTPUT} "=> Loading .*last_full\.pth" 2

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 2
check_checkpoints "${CHECKPOINTS}/stage3\.[[:digit:]]" 3
check_num_files ${CHECKPOINTS} 17   # 8x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


################################  pipeline 16  ################################
# testing on_stage_start option with different loading states and
# missing model state (should load best)
LOG_MSG='pipeline 16'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config16.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 4
check_line_counts ${EXP_OUTPUT} "=> Loading .*best_full\.pth" 2
check_line_counts ${EXP_OUTPUT} "=> Loading .*last_full\.pth" 2

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 2
check_checkpoints "${CHECKPOINTS}/stage3\.[[:digit:]]" 3
check_num_files ${CHECKPOINTS} 17   # 8x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


################################  pipeline 17  ################################
# testing on_stage_start option with missing on_stage_start and with
# dict with random key (test for ignoring random keys)
LOG_MSG='pipeline 17'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config17.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 1
check_line_counts ${EXP_OUTPUT} "=> Loading .*best\.pth" 1

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 2
check_num_files ${CHECKPOINTS} 11   # 5x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}


################################  pipeline 18  ################################
# testing loading from checkpoint specified by path
LOG_MSG='pipeline 18'
echo ${LOG_MSG}

LOGDIR=./tests/logs/_tests_dl_callbacks/for_resume
CHECKPOINTS=${LOGDIR}/checkpoints
LOGFILE=${CHECKPOINTS}/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config0.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 0

check_file_existence $LOGFILE
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_num_files $CHECKPOINTS 7   # 3x2 checkpoints + metrics.json

LOGDIR=./tests/logs/_tests_dl_callbacks/other
CHECKPOINTS=${LOGDIR}/checkpoints
LOGFILE=${CHECKPOINTS}/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config18.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 2
check_line_counts ${EXP_OUTPUT} "=> .*for_resume/checkpoints/best.pth" 1
check_line_counts ${EXP_OUTPUT} "=> .*for_resume/checkpoints/last_full.pth" 1

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

rm -rf ./tests/logs/_tests_dl_callbacks ${EXP_OUTPUT}


################################  pipeline 19  ################################
# testing loading from checkpoint specified by path in combination with other
LOG_MSG='pipeline 19'
echo ${LOG_MSG}

LOGDIR=./tests/logs/_tests_dl_callbacks/for_resume
CHECKPOINTS=${LOGDIR}/checkpoints
LOGFILE=${CHECKPOINTS}/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config0.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 0

check_file_existence $LOGFILE
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_num_files $CHECKPOINTS 7   # 3x2 checkpoints + metrics.json

LOGDIR=./tests/logs/_tests_dl_callbacks/other
CHECKPOINTS=${LOGDIR}/checkpoints
LOGFILE=${CHECKPOINTS}/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config19.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "=> Loading" 4
check_line_counts ${EXP_OUTPUT} "=> .*other/checkpoints/best_full.pth" 1
check_line_counts ${EXP_OUTPUT} "=> .*other/checkpoints/last_full.pth" 1
check_line_counts ${EXP_OUTPUT} "=> .*for_resume/checkpoints/best_full.pth" 1
check_line_counts ${EXP_OUTPUT} "=> .*for_resume/checkpoints/last_full.pth" 1

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 9   # 4x2 checkpoints + metrics.json

rm -rf ./tests/logs/_tests_dl_callbacks ${EXP_OUTPUT}


################################  pipeline 20  ################################
# testing loading from checkpoint specified by path in combination with other
#LOG_MSG='pipeline 20'
#echo ${LOG_MSG}
#
#LOGDIR=./tests/logs/_tests_dl_callbacks/trace
#CHECKPOINTS=${LOGDIR}/checkpoints
#TRACE=${LOGDIR}/trace
#LOGFILE=${CHECKPOINTS}/_metrics.json
#
#PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
#  python catalyst/dl/scripts/run.py \
#  --expdir=${EXPDIR} \
#  --config=${EXPDIR}/config20.yml \
#  --logdir=${LOGDIR} > ${EXP_OUTPUT}
#
#cat ${EXP_OUTPUT}
#
#check_file_existence ${LOGFILE}
#cat ${LOGFILE}
#echo ${LOG_MSG}
#
#check_checkpoints "${CHECKPOINTS}/best" 1
#check_checkpoints "${CHECKPOINTS}/last" 1
#check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
#check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + _metrics.json
#check_num_files ${TRACE} 1

rm -rf ./tests/logs/_tests_dl_callbacks ${EXP_OUTPUT}


################################  pipeline 21  ################################
# test CMCScoreCallback

echo 'pipeline 21'
rm -rf data/split_0
EXPDIR=./tests/_tests_ml_cmcscore
LOGDIR=./tests/logs/_tests_ml_cmcscore
LOGFILE=${LOGDIR}/checkpoints/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config1.yml \
  --logdir=${LOGDIR}

if [[ ! (-f "$LOGFILE" && -r "$LOGFILE") ]]; then
    echo "File $LOGFILE does not exist"
    exit 1
fi

cat $LOGFILE
echo 'pipeline 21'
python -c """
from catalyst import utils
import numpy as np
metrics = utils.load_config('$LOGFILE')

EPS = 0.00001
assert metrics['last']['cmc_1'] > 0.1  # slightly better then random
assert metrics['last']['cmc_5'] > 0.5
"""

rm -rf {LOGDIR}