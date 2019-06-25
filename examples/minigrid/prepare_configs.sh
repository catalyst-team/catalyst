#!/usr/bin/env bash

if [[ -z "$LOGDIR" ]]
then
      LOGDIR="."
fi

mkdir -p ${LOGDIR}/logs
mkdir -p ${LOGDIR}/mongodb

if [[ "$(uname)" == "Darwin" ]]; then
    sed -i ".bak" "s/logdir: .*/logdir: ${LOGDIR//\//\\/}\/logs\/minigrid-dqn/g" ./config_dqn.yml
    sed -i ".bak" "s/logdir: .*/logdir: ${LOGDIR//\//\\/}\/logs\/minigrid-ppo/g" ./config_ppo.yml
    sed -i ".bak" "s/path: .*/path: ${LOGDIR//\//\\/}\/mongo.log/g" ./mongod.conf
    sed -i ".bak" "s/dbPath: .*/dbPath: ${LOGDIR//\//\\/}\/mongodb/g" ./mongod.conf
elif [[ "$(expr substr $(uname -s) 1 5)" == "Linux" ]]; then
    sed -i "s/logdir: .*/logdir: ${LOGDIR//\//\\/}\/logs\/minigrid-dqn/g" ./config_dqn.yml
    sed -i "s/logdir: .*/logdir: ${LOGDIR//\//\\/}\/logs\/minigrid-ppo/g" ./config_ppo.yml
    sed -i "s/path: .*/path: ${LOGDIR//\//\\/}\/mongo.log/g" ./mongod.conf
    sed -i "s/dbPath: .*/dbPath: ${LOGDIR//\//\\/}\/mongodb/g" ./mongod.conf
fi
