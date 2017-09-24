export MXNET_CPU_WORKER_NTHREADS=48
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python2 fine-tune.py --pretrained-model model/resnet-152 \
    --load-epoch 0 --gpus 0,1,2,3 \
    --data-train data/scene_train_20170904.lst --model-prefix model/Scence-resnet-152 \
    --data-val data/scene_validation_20170908.lst \
	--data-nthreads 48 \
    --batch-size 16 --num-classes 80 --num-examples 53879

