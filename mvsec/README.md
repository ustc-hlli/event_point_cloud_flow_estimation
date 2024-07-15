Compute the velocity files of the MVSEC dataset.

The code is borrowed from [MVSEC](https://github.com/daniilidis-group/mvsec/tree/master/tools/gt_flow).

The code requires the rosbag library. Alternatively, you can directly use the computed velocity files in `./computed`. Copy them to the root of the MVSEC dataset before generating the scene flow and optical flow annotations.

## Usage
Run `compute_velocity.py` as follows.
```
python compute_velocity.py --mvsec_root MVSEC_ROOT --output OUTPUT
```

Prameters:

`--mvsec_root`: the root to the MVSEC ROS bag files (e.g. `outdoor_day1_gt.bag` and `outdoor_day2_gt.bag`).

`--output`: the path to save the computed velocity files.
