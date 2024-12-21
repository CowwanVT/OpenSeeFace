v4l2-ctl -d 2 -c zoom_absolute=500   #zoom in
v4l2-ctl -d 2 -c auto_exposure=3     #turn auto-exposure on
v4l2-ctl -d 2 -c exposure_dynamic_framerate=0    #this option is required for 30 fps
v4l2-ctl -d 2 -c sharpness=0
v4l2-ctl -d 2 -c gain=0
v4l2-ctl -d 2 -c contrast=128    #default value
v4l2-ctl -d 2 -c brightness=128 #default value
v4l2-ctl -d 2 -c focus_automatic_continuous=0 #auto-focus seemed to cause issues in low light
v4l2-ctl -d 2 -c focus_absolute=0
v4l2-ctl -d 2 -c power_line_frequency=0



#make sure the python virtual environment is created and activated
python3 -m venv .venv
source .venv/bin/activate

#the openseeface command
python3 facetracker.py  --fps 30 --ip 127.0.0.1 --model 3 -W 640 -H 480 -c "/dev/video2" #--preview 1 #-W 1920 -H 1080 #--preview 1

