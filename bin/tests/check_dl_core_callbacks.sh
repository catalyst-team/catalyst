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
    if [[ ! -f "$1" ]]
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
    check_num_files "${1}.pth" $2
    check_num_files "${1}_full.pth" $2
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

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

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

check_file_existence ${LOGFILE}
cat $LOGFILE;
echo ${LOG_MSG};

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
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

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 1
check_checkpoints "${CHECKPOINTS}/stage3\.[[:digit:]]" 1
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

check_file_existence ${LOGFILE}
cat $LOGFILE;
echo ${LOG_MSG};

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 3
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

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 3
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 3
check_checkpoints "${CHECKPOINTS}/stage3\.[[:digit:]]" 3
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

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
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

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
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

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
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

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 3
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 3
check_checkpoints "${CHECKPOINTS}/stage3\.[[:digit:]]" 3
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

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 1
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

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 3
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 3
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

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 9  # 4x2 checkpoints + metrics.json

rm -rf ${LOGDIR}

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
  --logdir=${LOGDIR}

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
  --logdir=${LOGDIR}

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR}


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
  --logdir=${LOGDIR}

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
  --logdir=${LOGDIR}

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR}

################################  pipeline 14  ################################
# testing on_stage_start option
LOG_MSG='pipeline 14'
echo ${LOG_MSG}

LOGDIR=./tests/logs/_tests_dl_callbacks
CHECKPOINTS=${LOGDIR}/checkpoints
LOGFILE=${CHECKPOINTS}/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config14.yml \
  --logdir=${LOGDIR}

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 2
check_checkpoints "${CHECKPOINTS}/stage3\.[[:digit:]]" 3
check_num_files ${CHECKPOINTS} 17   # 8x2 checkpoints + metrics.json

rm -rf ${LOGDIR}

################################  pipeline 15  ################################
# testing on_stage_start option with different loading states
LOG_MSG='pipeline 15'
echo ${LOG_MSG}

LOGDIR=./tests/logs/_tests_dl_callbacks
CHECKPOINTS=${LOGDIR}/checkpoints
LOGFILE=${CHECKPOINTS}/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config15.yml \
  --logdir=${LOGDIR}

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 2
check_checkpoints "${CHECKPOINTS}/stage3\.[[:digit:]]" 3
check_num_files ${CHECKPOINTS} 17   # 8x2 checkpoints + metrics.json

rm -rf ${LOGDIR}


################################  pipeline 16  ################################
# testing on_stage_start option with different loading states and
# missing model state (should load best)
LOG_MSG='pipeline 16'
echo ${LOG_MSG}

LOGDIR=./tests/logs/_tests_dl_callbacks
CHECKPOINTS=${LOGDIR}/checkpoints
LOGFILE=${CHECKPOINTS}/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config16.yml \
  --logdir=${LOGDIR}

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 2
check_checkpoints "${CHECKPOINTS}/stage3\.[[:digit:]]" 3
check_num_files ${CHECKPOINTS} 17   # 8x2 checkpoints + metrics.json

rm -rf ${LOGDIR}

################################  pipeline 17  ################################

LOG_MSG='pipeline 17'
echo ${LOG_MSG}

LOGDIR=./tests/logs/_tests_dl_callbacks
CHECKPOINTS=${LOGDIR}/checkpoints
LOGFILE=${CHECKPOINTS}/_metrics.json

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} 
  python3 -c "
import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst.dl import SupervisedRunner, State, Callback, CallbackOrder, CheckpointCallback

# experiment_setup
logdir = '${LOGDIR}'
num_epochs = 5

# data
num_samples, num_features = int(1e4), int(1e1)
X = torch.rand(num_samples, num_features)
y = torch.randint(0, 5, size=[num_samples])
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, num_workers=1)
loaders = {'train': loader, 'valid': loader}

# model, criterion, optimizer, scheduler
model = torch.nn.Linear(num_features, 5)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
runner = SupervisedRunner()

# first stage
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=False,
    callbacks=[
        CheckpointCallback(
            save_n_best=2,
            load_on_stage_end='best'
        ),
    ]
)
"

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/train\.[[:digit:]]" 2
check_num_files ${CHECKPOINTS} 9   # 4x2 checkpoints + metrics.json

rm -rf ${LOGDIR}


################################  pipeline 18  ################################

LOG_MSG='pipeline 18'
echo ${LOG_MSG}

LOGDIR=./tests/logs/_tests_dl_callbacks
CHECKPOINTS=${LOGDIR}/checkpoints
LOGFILE=${CHECKPOINTS}/_metrics.json

rm -rf ${LOGDIR}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} 
  python3 -c "
import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst.dl import SupervisedRunner, State, Callback, CallbackOrder, CheckpointCallback

# experiment_setup
logdir = '${LOGDIR}'
num_epochs = 5

# data
num_samples, num_features = int(1e4), int(1e1)
X = torch.rand(num_samples, num_features)
y = torch.randint(0, 5, size=[num_samples])
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, num_workers=1)
loaders = {'train': loader, 'valid': loader}

# model, criterion, optimizer, scheduler
model = torch.nn.Linear(num_features, 5)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
runner = SupervisedRunner()

# first stage
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=False,
    callbacks=[
        CheckpointCallback(
            save_n_best=2,
            load_on_stage_end={
                'model': 'best',
                'criterion': 'best',
                'optimizer': 'last',
            }
        ),
    ]
)
# second stage
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=False,
    callbacks=[
        CheckpointCallback(
            save_n_best=3,
            load_on_stage_start={
                'model': 'last',
                'criterion': 'last',
                'optimizer': 'best',
            }
        ),
    ]
)
"

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/train\.[[:digit:]]" 3
check_num_files ${CHECKPOINTS} 11   # 5x2 checkpoints + metrics.json

rm -rf ${LOGDIR}
