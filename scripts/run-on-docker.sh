#!/usr/bin/env bash
version=$1
Container="transformersx-"

function check_arguments()
{
    if [[ "${version}" == "" ]]; then
        version="turbo"
        echo "No Container input, try to use the default: ${Container}${version}"
    fi

    if ! test -e ./models; then
        echo "Please make soft link to add current user home for the models directory."
        exit 0
    fi

    if ! test -e ./dataset; then
        echo "Please make soft link to add current user home for the dataset directory."
        exit 0
    fi

    if ! docker images -f reference="${Container}${version}" | grep ${Container}${version}; then
        echo "No Docker image found for name ${Container},please build it firstly by using script 'build-docker.sh'"
        exit 0
    fi
}

function start()
{
    check_arguments

    parameters="run -it --rm --ipc=host --gpus all "
    cur_dir=`pwd`

    cd ./models
    models_dir=`pwd`
    cd $cur_dir

    cd ./dataset
    datasets_dir=`pwd`
    cd $cur_dir

    parameters="${parameters} -v ${models_dir}:/app/models "
    parameters="${parameters} -v ${datasets_dir}:/app/dataset "
    parameters="${parameters} -v ${cur_dir}:/app/nlp-workspace "

    echo "Start docker: ${parameters} ${Container}${version}"

    docker ${parameters} ${Container}${version}
}

start