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
# setup: run validation once in 3 epoch
LOG_MSG='pipeline 00'
echo ${LOG_MSG}

LOGDIR=./tests/logs/_tests_dl_callbacks
CHECKPOINTS=${LOGDIR}/checkpoints
LOGFILE=${CHECKPOINTS}/_metrics.json
EXP_OUTPUT=./tests/output.txt

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python3 -c "
import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst.dl import (
    SupervisedRunner, State, Callback, CallbackOrder,
    PeriodicLoaderCallback,
)

# experiment_setup
logdir = '${LOGDIR}'
num_epochs = 10

# data
num_samples, num_features = int(1e4), int(1e1)
X = torch.rand(num_samples, num_features)
y = torch.randint(0, 5, size=[num_samples])
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, num_workers=1)
loaders = {
    'train': loader,
    'valid': loader,
}

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
        PeriodicLoaderCallback(valid=3)
    ]
)
" > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "(train):" 10
check_line_counts ${EXP_OUTPUT} "(valid):" 3
check_line_counts ${EXP_OUTPUT} ".*/train\.[[:digit:]]\.pth" 1

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/train\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}

################################  pipeline 01  ################################
# setup: never run validation
LOG_MSG='pipeline 01'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python3 -c "
import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst.dl import (
    SupervisedRunner, State, Callback, CallbackOrder,
    PeriodicLoaderCallback,
)

# experiment_setup
logdir = '${LOGDIR}'
num_epochs = 5

# data
num_samples, num_features = int(1e4), int(1e1)
X = torch.rand(num_samples, num_features)
y = torch.randint(0, 5, size=[num_samples])
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, num_workers=1)
loaders = {
    'train': loader,
    'valid': loader,
}

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
        PeriodicLoaderCallback(valid=0)
    ]
)
" > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "(train):" 5
check_line_counts ${EXP_OUTPUT} "(valid):" 0
check_line_counts ${EXP_OUTPUT} ".*/train\.[[:digit:]]\.pth" 1

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/train\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}

################################  pipeline 02  ################################
# setup: multiple loaders with different periods
LOG_MSG='pipeline 02'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python3 -c "
import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst.dl import (
    SupervisedRunner, State, Callback, CallbackOrder,
    PeriodicLoaderCallback,
)

# experiment_setup
logdir = '${LOGDIR}'
num_epochs = 10

# data
num_samples, num_features = int(1e4), int(1e1)
X = torch.rand(num_samples, num_features)
y = torch.randint(0, 5, size=[num_samples])
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, num_workers=1)
loaders = {
    'train': loader,
    'train_additional': loader,
    'valid': loader,
    'valid_additional': loader,
}

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
        PeriodicLoaderCallback(
            train_additional=2,
            valid=3,
            valid_additional=0
        )
    ]
)
" > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "(train):" 10
check_line_counts ${EXP_OUTPUT} "(train_additional):" 5
check_line_counts ${EXP_OUTPUT} "(valid):" 3
check_line_counts ${EXP_OUTPUT} "(valid_additional):" 0
check_line_counts ${EXP_OUTPUT} ".*/train\.[[:digit:]]\.pth" 1

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/train\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}

################################  pipeline 03  ################################
# setup: multiple loaders with different periods with two stages
LOG_MSG='pipeline 03'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python3 -c "
import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst.dl import (
    SupervisedRunner, State, Callback, CallbackOrder,
    PeriodicLoaderCallback,
)

# experiment_setup
logdir = '${LOGDIR}'

# data
num_samples, num_features = int(1e4), int(1e1)
X = torch.rand(num_samples, num_features)
y = torch.randint(0, 5, size=[num_samples])
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, num_workers=1)
loaders = {
    'train': loader,
    'train_additional': loader,
    'valid': loader,
    'valid_additional': loader,
}

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
    num_epochs=5,
    verbose=False,
    callbacks=[
        PeriodicLoaderCallback(
            train_additional=2,
            valid=3,
            valid_additional=0
        )
    ]
)

# second stage
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir=logdir,
    num_epochs=10,
    verbose=False,
    callbacks=[
        PeriodicLoaderCallback(
            train_additional=2,
            valid=3,
            valid_additional=0
        )
    ]
)
" > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "(train):" 15
check_line_counts ${EXP_OUTPUT} "(train_additional):" 7
check_line_counts ${EXP_OUTPUT} "(valid):" 4
check_line_counts ${EXP_OUTPUT} "(valid_additional):" 0
check_line_counts ${EXP_OUTPUT} ".*/train\.[[:digit:]]\.pth" 2

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/train\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}

################################  pipeline 04  ################################
# setup: run validation once in 2 epoch
LOG_MSG='pipeline 04'
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
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}

################################  pipeline 05  ################################
# setup: never run validation
LOG_MSG='pipeline 05'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config1.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "(train):" 5
check_line_counts ${EXP_OUTPUT} "(valid):" 0
check_line_counts ${EXP_OUTPUT} ".*/stage1\.[[:digit:]]\.pth" 1

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}

################################  pipeline 05  ################################
# setup: multiple loaders
LOG_MSG='pipeline 05'
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
check_line_counts ${EXP_OUTPUT} ".*/stage1\.[[:digit:]]\.pth" 1

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}

################################  pipeline 06  ################################
# setup: multiple loaders and few stages
LOG_MSG='pipeline 06'
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
check_line_counts ${EXP_OUTPUT} ".*/stage2\.[[:digit:]]\.pth" 1

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_checkpoints "${CHECKPOINTS}/stage2\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 9   # 4x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}

# ###############################  pipeline 07  ################################
# setup: test for raising error when there is no loader in epoch

LOG_MSG='pipeline 07'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python3 -c "
import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst.dl import (
    SupervisedRunner, State, Callback, CallbackOrder,
    PeriodicLoaderCallback,
)

# experiment_setup
logdir = '${LOGDIR}'

# data
num_samples, num_features = int(1e4), int(1e1)
X = torch.rand(num_samples, num_features)
y = torch.randint(0, 5, size=[num_samples])
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, num_workers=1)
loaders = {
    'train': loader,
    'train_additional': loader,
    'valid': loader,
    'valid_additional': loader,
}

# model, criterion, optimizer, scheduler
model = torch.nn.Linear(num_features, 5)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
runner = SupervisedRunner()

try:
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=10,
        verbose=False,
        callbacks=[
            PeriodicLoaderCallback(
                train=2,
                train_additional=2,
                valid=3,
                valid_additional=0
            )
        ]
    )
except ValueError:
    print('Successfully handled error for epoch with no loaders!')
" > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

rm -rf ${LOGDIR} ${EXP_OUTPUT}

################################  pipeline 08  ################################
# setup: test for raising error when there is no loader in epoch

LOG_MSG='pipeline 08'
echo ${LOG_MSG}

{
    # try
    PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python catalyst/dl/scripts/run.py \
  --expdir=${EXPDIR} \
  --config=${EXPDIR}/config4.yml \
  --logdir=${LOGDIR} > ${EXP_OUTPUT}
} || {  # catch
    echo "Successfully handled error for epoch with no loaders!"
}

cat ${EXP_OUTPUT}

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

rm -rf ${LOGDIR} ${EXP_OUTPUT}

# ###############################  pipeline 09  ################################
# setup: test for ignoring some random loaders

LOG_MSG='pipeline 09'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python3 -c "
import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst.dl import (
    SupervisedRunner, State, Callback, CallbackOrder,
    PeriodicLoaderCallback,
)

# experiment_setup
logdir = '${LOGDIR}'

# data
num_samples, num_features = int(1e4), int(1e1)
X = torch.rand(num_samples, num_features)
y = torch.randint(0, 5, size=[num_samples])
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, num_workers=1)
loaders = {
    'train': loader,
    'train_additional': loader,
    'valid': loader,
    'valid_additional': loader,
}

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
    num_epochs=10,
    verbose=False,
    callbacks=[
        PeriodicLoaderCallback(
            train_additional=2,
            train_not_exists=2,
            valid=3,
            valid_additional=0,
            valid_not_exist=1,
        )
    ]
)
" > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}
check_line_counts ${EXP_OUTPUT} "(train):" 10
check_line_counts ${EXP_OUTPUT} "(train_additional):" 5
check_line_counts ${EXP_OUTPUT} "(train_not_exists):" 0
check_line_counts ${EXP_OUTPUT} "(valid):" 3
check_line_counts ${EXP_OUTPUT} "(valid_additional):" 0
check_line_counts ${EXP_OUTPUT} "(valid_not_exist):" 0
check_line_counts ${EXP_OUTPUT} ".*/train\.[[:digit:]]\.pth" 1

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/train\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}

################################  pipeline 10  ################################
# setup: test for ignoring some random loaders

LOG_MSG='pipeline 10'
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
check_line_counts ${EXP_OUTPUT} ".*/stage1\.[[:digit:]]\.pth" 1

check_file_existence ${LOGFILE}
cat ${LOGFILE}
echo ${LOG_MSG}

check_checkpoints "${CHECKPOINTS}/best" 1
check_checkpoints "${CHECKPOINTS}/last" 1
check_checkpoints "${CHECKPOINTS}/stage1\.[[:digit:]]" 1
check_num_files ${CHECKPOINTS} 7   # 3x2 checkpoints + metrics.json

rm -rf ${LOGDIR} ${EXP_OUTPUT}

# ###############################  pipeline 11  ################################
# setup: test for wrong type of period

LOG_MSG='pipeline 11'
echo ${LOG_MSG}

PYTHONPATH=./examples:./catalyst:${PYTHONPATH} \
  python3 -c "
import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst.dl import (
    SupervisedRunner, State, Callback, CallbackOrder,
    PeriodicLoaderCallback,
)

# experiment_setup
logdir = '${LOGDIR}'

# data
num_samples, num_features = int(1e4), int(1e1)
X = torch.rand(num_samples, num_features)
y = torch.randint(0, 5, size=[num_samples])
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, num_workers=1)
loaders = {
    'train': loader,
    'train_additional': loader,
    'valid': loader,
    'valid_additional': loader,
}

# model, criterion, optimizer, scheduler
model = torch.nn.Linear(num_features, 5)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
runner = SupervisedRunner()

try:
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=10,
        verbose=False,
        callbacks=[
            PeriodicLoaderCallback(
                train_additional=[],
                train_not_exists=2,
                valid=3,
                valid_additional=0,
                valid_not_exist=1,
            )
        ]
    )
except TypeError as e:
    print('Successfully handled type error for wrong period type!')
" > ${EXP_OUTPUT}

cat ${EXP_OUTPUT}

echo ${LOG_MSG}

rm -rf ${LOGDIR} ${EXP_OUTPUT}