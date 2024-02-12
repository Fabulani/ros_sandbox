# Notes for self

## Play rosbag

Run from root to play rosbag (`-l` to loop):

```
ros2 bag play ./data/standing_still/lidar_tf_tf_static_points_stationary_bag_1 -l

ros2 bag play ./data/moving/lidar_tf_tf_static_points_moving_bag_1 -l
```

Other useful arguments:
- `-d DELAY` to apply a delay in seconds before play.
- `-p` to start paused.
- `--clock [CLOCK]` publish to /clock at a specific frequency in Hz.
- `-r RATE` rate at which to play back messages.


## Open rviz2

`rviz2`

For `/PointCloud2`:
- change `Topic/Reliability Policy` to `Best Effort`

For `/tf`:
- change `Frame Timeout` to a high number so the tf doesn't disappear.


## About parser.py

- For each point in PointCloud2:
    - if zero3, either barrier hit or inf
    - closer to 30cm to the lidar -> blocked (zero3)
    - farther than 50m -> inf (zero3)