#! /bin/bash

# Usage ./replicate_results.sh <DATASET-NAME> <THREAD-ID> <JE>
## where <DATASET-NAME>: MNIST, CIFAR

DATASET=$1
THREAD=$2
GPU_ID=$3
JE=$4
m1=$5
m2=$6
s=$7
OPTIM='SGD'
BATCH_SIZE=10
if [ $DATASET = "MNIST" ]; then
    NUM_RUNS=5
    if [ $THREAD = 1 ]; then
        IMP_METHODS=( 'VAN' 'EWC' 'PI' )
        LRS=(0.03 0.03 0.1)
        LAMDAS=(0 10 0.1)
    elif [ $THREAD = 2 ]; then
        IMP_METHODS=( 'MAS' 'RWALK' )
        LRS=(0.1 0.1)
        LAMDAS=(0.1 1)
    elif [ $THREAD = 3 ]; then
        #IMP_METHODS=( 'PNN' 'A-GEM' )
        IMP_METHODS=( 'A-GEM')
        LRS=(0.1 0.1)
        LAMDAS=(0 0)
    elif [ $THREAD = 4 ]; then
        #IMP_METHODS=( 'PNN' 'A-GEM' )
        IMP_METHODS=( 'MEGA')
        LRS=(0.1 0.1)
        LAMDAS=(0 0)
    elif [ $THREAD = 5 ]; then
        #IMP_METHODS=( 'PNN' 'A-GEM' )
        IMP_METHODS=( 'MEGAD')
        LRS=(0.1 0.1)
        LAMDAS=(0 0)
    elif [ $THREAD = 6 ]; then
        #IMP_METHODS=( 'PNN' 'A-GEM' )
        IMP_METHODS=( 'AKCL')
        LRS=(0.1 0.1)
        LAMDAS=(0 0)
    fi
    ARCH='FC-S'
    RESULTS_DIR='results/mnist'
    if [ ! -d $RESULTS_DIR ]; then
        mkdir -pv $RESULTS_DIR
    fi

    for ((i=0;i<${#IMP_METHODS[@]};++i)); do
        imp_method="${IMP_METHODS[i]}"
        lr=${LRS[i]}
        lam=${LAMDAS[i]}
        CUDA_VISIBLE_DEVICES=$GPU_ID python ./fc_permute_mnist.py --train-single-epoch --arch $ARCH --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --learning-rate $lr --imp-method $imp_method --synap-stgth $lam --log-dir $RESULTS_DIR --m1-value $5 --m2-value $6 --s-value $7
    done
elif [ $DATASET = "CIFAR" ]; then
    NUM_RUNS=1
    if [ $THREAD = 1 ]; then
        IMP_METHODS=( 'VAN' 'EWC' 'PI' )
        LRS=(0.01 0.03 0.01)
        LAMDAS=(0 10 0.1)
    elif [ $THREAD = 2 ]; then
        IMP_METHODS=( 'MAS' 'RWALK' )
        LRS=(0.03 0.03)
        LAMDAS=(0.1 1)
    elif [ $THREAD = 3 ]; then
        #IMP_METHODS=( 'PNN' 'A-GEM' )
        IMP_METHODS=( 'A-GEM' )
        LRS=(0.03 0.03)
        LAMDAS=(0 0)
    elif [ $THREAD = 4 ]; then
        #IMP_METHODS=( 'MEGAD' )
        IMP_METHODS=( 'MEGA' )
        LRS=(0.03 0.03)
        LAMDAS=(0 0)
    elif [ $THREAD = 5 ]; then
        IMP_METHODS=( 'AKCL' )
        LRS=(0.03 0.03)
        LAMDAS=(0 0)
    fi
    ARCH='RESNET-S'
    RESULTS_DIR='results/cifar'
    if [ ! -d $RESULTS_DIR ]; then
        mkdir -pv $RESULTS_DIR
    fi

    for ((i=0;i<${#IMP_METHODS[@]};++i)); do
        imp_method="${IMP_METHODS[i]}"
        lr=${LRS[i]}
        lam=${LAMDAS[i]}
        CUDA_VISIBLE_DEVICES=$GPU_ID python ./conv_split_cifar.py --train-single-epoch --arch $ARCH --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --learning-rate $lr --imp-method $imp_method --synap-stgth $lam --log-dir $RESULTS_DIR
    done
fi
