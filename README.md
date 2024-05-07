# A fork of OpenSeeFace

This is kind of a personal project, but it spiraled out of control and I figured I'd share it in case anyone else can make use of it. I make no promises to the quality of this code or the reliablity of it and I am not responsible if this somehow causes you issues.

I have drastically restructured this, but I'm listing it as a fork because this is still based on emilianavt's work. 

I now exclusively target Vtube Studio, all communication is done over the VTS API.

For more technical explainations on the face tracking and computer vision models see EmilianaVT's documentation https://github.com/emilianavt/OpenSeeFace

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

## Changes from OpenSeeFace
* Major restructure  
  * Separated major functions into discrete files
  * Larger focus on use of objects
* Webcam and VTS communication are now done in separate threads
* VTS communication is now done via API
* Added some Vbridger parameters
* Added brightness correction to webcam input, which should help with low light conditions
* Changed parameter calculations
  * Normalization has been changed to a completely new model
  * Added response curves to parameters
  * Parameters now use Eucledian distance instead of X/Y distances
  * All VTS values are now treated as parameters
  * Completely changed parameter calculations
* Added functionality to mitigate errant eye movements
* Reduced camera latency
* Main thread now skips late frames
* Reduced the threads used by OpenCV
* Various optimizations

## License 

### For the original code I modified and the computer vision models:

BSD 2-Clause License

Copyright (c) 2019, Emiliana (https://twitter.com/Emiliana_vt / https://github.com/emilianavt/)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

### For the parts of the code written/modified by me:

BSD 2-Clause License

Copyright (c) 2024, Cowwan (https://github.com/CowwanVT/ https://bsky.app/profile/cowwan.bsky.social https://www.twitch.tv/cowwan)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.


idk, I didn't plan to get this far
feel free to use my software and code for whatever, it'd be nice to get credit if you do
Just don't do anything to get me sued, or harass people, or make hateful content
