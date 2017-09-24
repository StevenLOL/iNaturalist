# iNaturalist
Originally written by phunterlau
<https://github.com/phunterlau/iNaturalist>.
MXNet fine-tune baseline script (resnet 152 layers) for iNaturalist Challenge at FGVC 2017, public LB score 0.117 from a single 21st epoch submission without ensemble.
Small changes made to use for <https://challenger.ai/competition/scene>

## How to use

### Install MXNet 

Run `pip install mxnet-cu80` after installing CUDA driver or go to <https://github.com/dmlc/mxnet/> for the latest version from Github.

Windows users? no CUDA 8.0? no GPU? Please run `pip search mxnet` and find the good package for your platform.

### Generate lists

After downloading and unzipping the train and validation sets. Rename images folder as 'scene_validation_20170908' and
'scene_train_20170904'. Rename anotation as
   'scene_validation_20170908.json'
'scene_train_20170904.json'. Then python2 mk_list.py will create two lists  'scene_validation_20170908.lst'
'scene_train_20170904.lst'.



### Train

Run `sh run.sh` which looks like (a 4 GTX 1080 machine for example):

```
export MXNET_CPU_WORKER_NTHREADS=48
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python2 fine-tune.py --pretrained-model model/resnet-152 \
    --load-epoch 0 --gpus 0,1,2,3 \
    --data-train data/scene_train_20170904.lst --model-prefix model/Scence-resnet-152 \
    --data-val data/scene_validation_20170908.lst \
	--data-nthreads 48 \
    --batch-size 16 --num-classes 80 --num-examples 53879

```

please adjust `--gpus` and `--batch-size` according to the machine configuration. A sample calculation: `batch-size = 12` can use 8 GB memory on a GTX 1080, so `--batch-size 48` is good for a 4-GPU machine.

Please have internet connection for the first time run because needs to download the pretrained model from <http://data.mxnet.io/models/imagenet-11k/resnet-152/>. If the machine has no internet connection, please download the corresponding model files from other machines, and ship to `model/` directory.

### Generate submission file

After a long run of some epochs, e.g. 30 epochs, we can select some epochs for the submission file. Run `sub.py` which two parameters : `num of epoch` and `gpu id` like:

```
python sub.py 12 1
```

selects the 12st epoch and infer on GPU `#1`. One can merge multiple epoch results on different GPUs and ensemble for a good submission file.

## Validation 
From the evaluation scipts given here<https://github.com/AIChallenger/AI_Challenger/tree/master/Evaluation/scene_classification_eval>
'''
python2 scene_eval.py --submit submit.json --ref  ./data/scene_validation_20170908.json Evaluation time of your result: 3.540856 s
{'warning': [], 'score': '0.953230337079', 'error': []}
'''
## How 'fine-tune' works

Fine-tune method starts with loading a pretrained ResNet 152 layers (Imagenet 11k classes) from MXNet model zoo, where the model has gained some prediction power, and applies the new data by learning from provided data. 

The key technique is from `lr_step_epochs` where we assign a small learning rate and less regularizations when approach to certain epochs. In this example, we give `lr_step_epochs='10,20'` which means the learning rate changes slower when approach to 10th and 20th epoch, so the fine-tune procedure can converge the network and learn from the provided new samples. A similar thought is applied to the data augmentations where fine tune is given less augmentation. This technique is described in Mu's thesis <http://www.cs.cmu.edu/~muli/file/mu-thesis.pdf> 

This pipeline is not limited to ResNet-152 pretrained model. Please experiment the fine tune method with other models, like ResNet 101, Inception, from MXNet's model zoo <http://data.mxnet.io/models/> by following this tutorial <http://mxnet.io/how_to/finetune.html> and this sample code <https://github.com/dmlc/mxnet/blob/master/example/image-classification/fine-tune.py> . Please feel free submit issues and/or pull requests and/or discuss on the Kaggle forum if have better results.

## Reference

* MXNet's model zoo <http://data.mxnet.io/models/>
* MXNet fine tune <http://mxnet.io/how_to/finetune.html> <https://github.com/dmlc/mxnet/blob/master/example/image-classification/fine-tune.py>
* Mu Li's thesis <http://www.cs.cmu.edu/~muli/file/mu-thesis.pdf> 
* iNaturalist Challenge at FGVC 2017 <https://www.kaggle.com/c/inaturalist-challenge-at-fgvc-2017/>
