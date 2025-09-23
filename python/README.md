# Optical Flow Ball (Python)

This is the Python implementation of the Optical Flow Ball demo.  
It uses OpenCV (cv2) to track feature points with the Lucas-Kanade method.

## Setup
Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python app.py
```
## Features
Tracks good feature points in feed.
Draws decaying trails to visualize motion.
Displays live stats on tracked featues and average motion.
ball interacts with motion globally (not localized).