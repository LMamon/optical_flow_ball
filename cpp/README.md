# Optical Flow Ball (C++)

This folder contains the C++ implementation of the Optical Flow Ball demo.  
It extends the Python version by localizing motion interaction:  
the ball reacts only to motion detected within its region of interest (ROI), creating a more responsive and game-like interaction.


## Setup

### Dependencies
- **C++17 compiler** (e.g., `clang++` or `g++`)
- **CMake 3.10+**
- **OpenCV 4.x** (installed system-wide, e.g., via `brew install opencv` or `apt install libopencv-dev`)

### Build
```bash
# from repo root
mkdir build
cd build
cmake ../cpp
make

./app
```

## Features
Tracks optical flow within a rectangle centered on the ball.
Average motion vectors inside the ROI are applied to the ball's velocity.
Gravity and wall collisions are simulated
the balls motion is influenced by local gestures or movement captured in the feed.