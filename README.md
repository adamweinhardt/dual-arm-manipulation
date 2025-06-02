# dual-arm-manipulation

## Run simulation:

```bash
docker run --rm -it \
  -p 6080:6080 -p 5901:5900 -p 29999:29999 \
  -p 30001-30004:30001-30004 -p 30020:30020 \
  -p 50001-50003:50001-50003 -p 5002:502 \
  -e ROBOT_MODEL=UR5 \
  docker.io/universalrobots/ursim_e-series
```

