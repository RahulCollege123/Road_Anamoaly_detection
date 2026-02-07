#!/bin/bash

# wait for system + camera
sleep 15

# GUI env
export DISPLAY=:0
export XAUTHORITY=/home/pi/.Xauthority

# go to project
cd /home/pi/road_anomaly

# RUN USING VENV PYTHON DIRECTLY
/home/pi/edgeai/bin/python detect_live.py >> /home/pi/road_anomaly/boot_log.txt 2>&1
