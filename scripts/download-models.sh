#!/usr/bin/env bash

MODELS_DIR=$1

docker_model_prefix="registry.cn-beijing.aliyuncs.com/modoso/transformersx-"
model_types=("bert" "albert" "roberta" "electra")
cur_dir=`pwd`

function model_pull(){
    for type in ${model_types[*]}
    do
        if ! docker images -f reference="${docker_model_prefix}${type}" | grep ${docker_model_prefix}${type}; then
            echo "Pulling ${docker_model_prefix}${type}"
            docker pull ${docker_model_prefix}${type}
        fi
    done
}

function model_copy(){
    for type in ${model_types[*]}
    do
        target_image=${docker_model_prefix}${type}
        echo "Trying to copy models from docker image: ${target_image}:/app/models into ${MODELS_DIR}"
        docker run --rm -v ${MODELS_DIR}:/app/transformers ${target_image} sh -c "cp -r /app/models/* /app/transformers"
    done
}

function start() {
    if [[ "${MODELS_DIR}" = "" ]]; then
        MODELS_DIR=`echo ~/models/transformers/pretrained`
        echo "No directory specified for the models, ${MODELS_DIR} will be used."
    fi
    ! test -e ${MODELS_DIR} && mkdir -p ${MODELS_DIR}
    model_pull && model_copy
}

start

