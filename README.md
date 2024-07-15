# event_point_cloud_flow_estimation
Code for "Joint Flow Estimation from Point Clouds and Event Streams". [ICME 2024]

## Preparation
First, install the PointNet2 cpp lib as follows.
```
cd pointnet2
python setup.py install
cd ..
```
Then, we need to prepare the datasets.

For DSEC, we use the official trainig set for experiments. Download `train_events.zip`, `train_disparity.zip`, `train_optical_flow.zip`, and `train_calibration.zip` from the [official web page](https://dsec.ifi.uzh.ch/dsec-datasets/download/).

Unzip the files in `DSEC_ROOT` and then run `data_dsec.py` to process the dataset.
```
python data_dsec.py --dsec_root DSEC_ROOT --output OUTPUT
```
The processed DSEC dataset is stored in `OUTPUT`.

For MVSEC, we use the outdoor_day1 and outdoor_day2 sequences for experiments. Download the corresponding calibration files, HDF5 files and ROS bags from the [official web page](https://daniilidis-group.github.io/mvsec/download/) and place them in `MVSEC_ROOT`.

Before generatin the scene flow and optical flow annotations, we need to compute the 
## Evaluation
## Training
## Acknowledgement
