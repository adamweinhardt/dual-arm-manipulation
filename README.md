# UR5 simulator:

```bash
docker run --rm -it \
  -p 6080:6080 -p 5901:5900 -p 29999:29999 \
  -p 30001-30004:30001-30004 -p 30020:30020 \
  -p 50001-50003:50001-50003 -p 5002:502 \
  -e ROBOT_MODEL=UR5 \
  docker.io/universalrobots/ursim_e-series
```

# Robot IPC Control:

```bash
cd robot_ipc_control/
```

### Camera calibration:

```python3
python3 pose_estimation/camera_calibration.py --serial_number 138322253845
```

### Robot base calibration:

```bash
./controller/build/impedance_controller controller/config_right.json
```

```python3
python3 pose_estimation/robot_base_calibration.py --name right --robot_config_path controller/config_right.json -s 138322253845
```

```bash
./controller/build/impedance_controller controller/config_left.json
```

```python3
python3 pose_estimation/robot_base_calibration.py --name left --robot_config_path controller/config_left.json -s 138322253845
```

### Pose estimation:

```python3
python3 pose_estimation/pose_estimator.py --config=configs/pose_estimation_config_single_camera_dual_arm.json --detection_type=ransac_with_refinement
```

# Dual arm manipulation:

### Visualisation:

```python3
python3 visualizer/visualizer.py
```

### Control:

```python3
python3 control/dual_arm_controller.py
```

### Reset for mac:
```bash
sudo networksetup -setdhcp "USB 10/100/1000 LAN"
```
