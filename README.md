# [Super Resolution] SRDenseNet - tensorflow implementation
tensorflow implementation of SRDenseNet

## Prerequisites
 * python 3.x
 * Tensorflow > 1.x
 * Pillow
 * OpenCV
 * argparse

## Properties (what's different from reference code)
 * This code requires Tensorflow. This code was fully implemented based on Python 3 differently from the original.
 * This code supports both RGB and YCBCR channel space.
 * This code supports tensorboard summarization
 * This code supports model saving and restoration
 * This code uses PILLOW library to resize image. Note that the performance of Bicubic function in PILLOW is lower than that of Matlab library. 
 * Inference code should be changed to suit your environment.
 * This code is suitable for practical usage rather than research.
 * We use Resize-convolution instead Transpose-convolution. It is more flexible to handle various scale factor. 


## Usage
```
usage: python3 trainer.py --gpu 0 

[options]
""" system """
parser.add_argument("--exp_type", type=int, default=0, help='experiment type')
parser.add_argument("--gpu", type=str, default=1)  # -1 for CPU
parser.add_argument("--model_tag", type=str, default="default", help='Exp name to save logs/checkpoints.')
parser.add_argument("--checkpoint_dir", type=str, default='../__outputs/checkpoints/', help='Dir for checkpoints.')
parser.add_argument("--summary_dir", type=str, default='../__outputs/summaries/', help='Dir for tensorboard logs.')
parser.add_argument("--restore_model_file", type=str, default=None, help='file for restoration')
#parser.add_argument("--restore_model_file", type=str, default='../__outputs/checkpoints/SRDenseNet_SRDenseNet_model_default_10_11_00_55_14/model.ckpt-47000', help='file for resotration')
""" model """
parser.add_argument("--batch_size", type=int, default=16, help='Minibatch size(global)')
parser.add_argument("--patch_size", type=int, default=54, help='Minipatch size(global)')
#parser.add_argument("--patch_stride", type=int, default=13, help='patch stride') #we just sample patches randomly for simplicity
parser.add_argument("--operating_channel", type=str, default="RGB", help="operating channel [RGB, YCBCR")  # -1 for CPU
parser.add_argument("--num_channels", type=int, default=3, help='the number of channels')
parser.add_argument("--scale", type=int, default=3, help='scaling factor')
parser.add_argument("--data_root_train", type=str, default="./dataset/SR_training_datasets/T91", help='Data root dir')
parser.add_argument("--data_root_test", type=str, default="./dataset/SR_testing_datasets/Set5", help='Data root dir')
```

 * For running tensorboard, `tensorboard --logdir=../__outputs/summaries` then access localhost:6006 with your browser

## Result [Bicubic / SRDenseNet / Label (x3)]
<p align="center">
<img src="https://github.com/ppooiiuuyh/assets/blob/master/SRDense_1.png?raw=true" width="600">
</p>

<p align="center">
<img src="https://github.com/ppooiiuuyh/assets/blob/master/SRDense_2.png?raw=true" width="600">
</p>

<p align="center">
<img src="https://github.com/ppooiiuuyh/assets/blob/master/SRDense_3.png?raw=true" width="600">
</p>

<p align="center">
<img src="https://github.com/ppooiiuuyh/assets/blob/master/SRDense_4.png?raw=true" width="600">
</p>




## References
* [SRDenseNet](http://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf) : reference paper
* [kweisamx/TensorFlow-SR-DenseNet](https://github.com/kweisamx/TensorFlow-SR-DenseNet) : reference implementation
* [taki0112/Tensorflow-cookbook](https://github.com/taki0112/Tensorflow-Cookbook) : useful tensorflow cook reference

## Author
Dohyun Kim

