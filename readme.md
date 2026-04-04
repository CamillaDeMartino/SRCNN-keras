# Keras implementation of SRCNN


The original paper is [Learning a Deep Convolutional Network for Image Super-Resolution](https://arxiv.org/abs/1501.00092)

<p align="center">
  <img src="https://github.com/MarkPrecursor/SRCNN-keras/blob/master/SRCNN.png" width="800"/>
</p>

My implementation have some difference with the original paper, include:

* use Adam alghorithm for optimization, with learning rate 0.0003 for all layers.
* Use the opencv library to produce the training data and test data, not the matlab library. This difference may caused some deteriorate on the final results.
* I did not set different learning rate in different layer, but I found this network still work.
* The color space of YCrCb in Matlab and OpenCV also have some difference. So if you want to compare your results with some academic paper, you may want to use the code written with matlab.

## Use:
### Create your own data
Put your RGB training images in `data/raw/RGB/` or update the fallback path in `srcnn/prepare_data.py`.

Excute:
`python -m srcnn.prepare_data`

### training and test:
Excute:
`python -m srcnn.main`

## Project layout:
* `srcnn/` - source code
* `data/` - training, test and processed datasets
* `assets/` - example images used by the demo scripts
* `weights/` - model checkpoints and weights
* `outputs/` - generated predictions and intermediate results


## Result(training for 200 epoches on 91 images, with upscaling factor 2):
Results on Set5 dataset:
<p align="center">
  <img src="https://github.com/MarkPrecursor/SRCNN-keras/blob/master/result.png" width="800"/>
</p>


