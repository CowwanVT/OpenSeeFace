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

that's the setup complete, now you can either run "./facetracking.sh" to use my preferred settings, or "python3 facetracker.py" followed by your preferred command line arguments



### Windows:

* install python (be sure to check the box to add it to the environment variables)
* Download and install the Visual C++ 2019 runtime
* download, extract, and navigate to the source code in CMD or powershell
* run "py -m pip install numpy opencv-python onnxruntime pillow websockets"

setup is now complete, you can run the face tracker with "py facetracker.py", unfortunately I don't use windows and don't have a handy .bat file to handle everything




I'll update this readme to explain more later, but first I need to actually upload all my changes. I didn't expect this to turn into something I wanted to upload. 

For documentation on how openseeface works and the computer vision models, go see https://github.com/emilianavt/OpenSeeFace
I don't want to leave documentation here that makes it seem like this is all my work


Seriously, emilianavt did all the hard work


actual documentation is in the README.txt because working in plain text is easier than making github's markdown cooperate
