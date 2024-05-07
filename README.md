# A fork of OpenSeeFace

This is kind of a personal project, but it spiraled out of control and I figured I'd share it in case anyone else can make use of it. I make no promises to the quality of this code or the reliablity of it and I am not responsible if this somehow causes you issues.

I have drastically restructured this, but I'm listing it as a fork because this is still based on emilianavt's work. 

I now specifically target Vtube Studio, all communication is done over the VTS API. 

## How to use:

### Linux:

* install python, virtualenv, and pip
* download the source code (there is no release, it's just python scripts)
* extract the source code
* open a terminal in the directory with the source code
* run "python3 -m venv .venv" then "source .venv/bin/activate" to create a virtual environment and activate it
* run "pip install numpy opencv-python onnxruntime pillow websockets"

that's the setup complete, now you can run the face tracker with "python3 facetracker.py" followed by your preferred command line arguments



### Windows:

* install python (be sure to check the box to add it to the environment variables)
* Download and install the Visual C++ 2019 runtime
* download, extract, and navigate to the source code in CMD or powershell
* run "py -m pip install numpy opencv-python onnxruntime pillow websockets"

setup is now complete, you can run the face tracker with "py facetracker.py" followed by any command line arguments you need

## Command Line Arguments
* `-i` or `--ip` sets the target IP if it's not the local machine for some reason
* `-a` or `--api-port` sets the target port for the VTS api
* `-W` or `--width` sets the webcam capture width, defaults to 640 (Note: going above 640x480 can cause the webcam latency to go higher without much benefit)
* `-H` or `--height` sets the webcam capture height
* `-F` or `--fps` sets the webcam fps and target tracking fps. Defaults to 24 just to be safe, 30 should be fine for most webcams. Setting this to 60 is inadvisable because I've never managed to keep my frame times consistently below 16.6666ms while testing, but it might work with `--threads 2`
* `-c` or `--capture` sets the camera ID, mostly used for when you have multiple webcams
* `-M` or `--mirror-imput` flips the webcam input
* `-t` or `--threshold` sets the minimum confidence threshold for face tracking, valid values are decimals between 0 and 1, defaults to 0.7
* `-d` or `--detection-threshold` sets the minimum confidence threshold for face detection, valid values are decibals between 0 and 1, defaults to 0.7
* `-s` or `--silent` disables the stats printed to the console every frame (currently not implemented)
* `--model` sets the tracking model, valid values are 0,1,2,3,4 defaults to 3. Model 4 has better winking, models 0-2 are faster but less accurate
* `--preview` when set to 1 shows the webcam output pre-tracking, useful for tracking down tracking issues
* `-T` or `--threads` sets the number of threads used by the landmark detection model. Default is 1, 2 seems to get a decent speedup at the cost of higher cpu usage. Values higher than 2 don't seem to have any effect.
* `-v` or `--visualize` setting this to 1 enables a preview with a landmarks displayed, probably broken at the moment
* `--target-brightness` sets the garget brightness for gamma correction. Values are decimals between 0.25 and 0.75, default is 0.55


I'll update this readme to explain more later, but first I need to actually upload all my changes. I didn't expect this to turn into something I wanted to upload. 

For documentation on how openseeface works and the computer vision models, go see https://github.com/emilianavt/OpenSeeFace
I don't want to leave documentation here that makes it seem like this is all my work


Seriously, emilianavt did all the hard work


actual documentation is in the README.txt because working in plain text is easier than making github's markdown cooperate
