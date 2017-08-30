# Semantic Segmentation
### Introduction

In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Notes

The architecture is essentially as described in lecture. I was not sure whether `vgg_layer7_out` already included a 1x1 convolution, so I allowed the length of the layer's 1x1 convolution to be zero in order to omit it.

The hyperparameters and that aspect of the architecture were chosen following two grid searches. The code for the grid search is in the `results.ipynb` notebook. It splits the training set 80/20 into training and validation sets and records the loss and mean Intersection-Over-Union for the validation set. The top results are selected based on the validation loss.

All training was on a p2.xlarge (Tesla K80). The batch size of 13 was selected to avoid running out of memory on the GPU; much larger batches resulted in errors, and `nvidia-smi` reported memory usage fairly close to full for batches of 13.

#### Before Adding Augmented Data

I ran a grid search over
```
params_dict = {
        'batch_size': [13],
        'min_epochs_without_progress': [3],
        'max_epochs': [50],
        'keep_prob': [0.5],
        'learning_rate': [0.001, 0.0001, 0.00001],
        'kernel_size_3': [8, 16],
        'kernel_size_4': [2, 4],
        'kernel_size_7': [2, 4],
        'conv_1x1_depth': [0, 2048, 4096]
    }
```

The top 3 results were
```
[({'best_epoch': 25,
   'best_mean_iou': 0.82627001960398783,
   'best_validation_loss': 0.096971082737890346,
   'time': 1344.4162635760001},
  {'batch_size': 13,
   'conv_1x1_depth': 2048,
   'keep_prob': 0.5,
   'kernel_size_3': 8,
   'kernel_size_4': 2,
   'kernel_size_7': 2,
   'learning_rate': 0.001,
   'max_epochs': 50,
   'max_epochs_without_progress': 3}),
 ({'best_epoch': 18,
   'best_mean_iou': 0.83097683272119294,
   'best_validation_loss': 0.10434014350175858,
   'time': 1115.9451880589995},
  {'batch_size': 13,
   'conv_1x1_depth': 2048,
   'keep_prob': 0.5,
   'kernel_size_3': 8,
   'kernel_size_4': 2,
   'kernel_size_7': 4,
   'learning_rate': 0.001,
   'max_epochs': 50,
   'max_epochs_without_progress': 3}),
 ({'best_epoch': 19,
   'best_mean_iou': 0.83064380742735777,
   'best_validation_loss': 0.10534888960547366,
   'time': 1106.4235965560001},
  {'batch_size': 13,
   'conv_1x1_depth': 0,
   'keep_prob': 0.5,
   'kernel_size_3': 8,
   'kernel_size_4': 4,
   'kernel_size_7': 2,
   'learning_rate': 0.001,
   'max_epochs': 50,
   'max_epochs_without_progress': 3}),
```

The results on the test set for the top selection were not bad. Many of the mistakes involved shadows, which motivated some augmentation.

#### With Augmented Data

The transformations were;
- flip the image horizontally
- darken it with gamma correction
- lighten it with gamma correction

The grid was the same, except that I omitted the smallest learning rate, since it was very slow and did not score well in the first grid.

The top 3 were:
```
({'best_epoch': 16,
   'best_mean_iou': 0.87002838068994981,
   'best_validation_loss': 0.081844574076005788,
   'time': 5141.349783108002},
  {'batch_size': 13,
   'conv_1x1_depth': 0,
   'keep_prob': 0.5,
   'kernel_size_3': 8,
   'kernel_size_4': 4,
   'kernel_size_7': 4,
   'learning_rate': 0.001,
   'max_epochs': 50,
   'max_epochs_without_progress': 3}),
({'best_epoch': 9,
  'best_mean_iou': 0.85317735699401509,
  'best_validation_loss': 0.086172087010981016,
  'time': 3429.2166186340037},
 {'batch_size': 13,
  'conv_1x1_depth': 2048,
  'keep_prob': 0.5,
  'kernel_size_3': 8,
  'kernel_size_4': 4,
  'kernel_size_7': 4,
  'learning_rate': 0.001,
  'max_epochs': 50,
  'max_epochs_without_progress': 3}),   
({'best_epoch': 13,
  'best_mean_iou': 0.86400561086062733,
  'best_validation_loss': 0.090822377937963633,
  'time': 4377.303030897994},
 {'batch_size': 13,
  'conv_1x1_depth': 2048,
  'keep_prob': 0.5,
  'kernel_size_3': 8,
  'kernel_size_4': 2,
  'kernel_size_7': 4,
  'learning_rate': 0.001,
  'max_epochs': 50,
  'max_epochs_without_progress': 3}),
```

So, with augmentation, we achieved a higher (better) IOU, and used slightly larger kernels on the 4 and 7 transposed convolution layers (more parameters). The number of training epochs before the onset of overfitting was also reduced.

### Setup

```
ssh ubuntu@...
wget http://kitti.is.tue.mpg.de/kitti/data_road.zip
unzip data_road.zip
mv data_road data
mv data_road.zip ..

lsblk
sudo mkdir /data
sudo mount /dev/xvdf /data/
sudo mount /dev/xvdba /data/
conda install -c conda-forge tqdm

cd /data/CarND-Semantic-Segmentation
jupyter notebook --generate-config
vi /home/ubuntu/.jupyter/jupyter_notebook_config.py
edit to set `c.NotebookApp.ip = '*'`
jupyter notebook
copy URL and paste host name
```


```
grid_1: CPU training for a couple of settings to test

Example training output:
{'min_epochs_without_progress': 3, 'kernel_size_4': 4, 'max_epochs': 10, 'batch_size': 13, 'keep_prob': 0.5, 'conv_1x1_depth': 0, 'learning_rate': 0.0001, 'kernel_size_3': 16, 'kernel_size_7': 4}
INFO:tensorflow:Restoring parameters from b'./data/vgg/variables/variables'
```

```
grid_2: GPU training on a p2.xlarge (Tesla K80)
note: min_epochs_... should be max_epochs_...

full trained and tested in
29m55.069s

results in runs/1503612139.8536005
pretty good; passing; some challenges on shadows
maybe try augmentation
```

```
grid_3: GPU training on a p2.xlarge (Tesla K80)
- with augmentation (5 additional examples per training example)
- but augmentation on both training and validation sets... losses quite low,
suggesting overfitting
- partial run; best solutions identified were




grid_4: like grid_3, but with a 70/30 split and augmentation only on the
training data (not validation data)

Best solutions:


main run took 118m9.826s on the first result
71m24.706s on the second
```

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder
