# Optical Flow Ball

This demo/project explores interactive motion tracking using optical flow.  
It demonstrates two implementations — one in Python and one in C++ to compare approaches across languages and APIs.

## Purpose
- apply optical flow methods (Lucas-Kanade method).
- Build an interactive demo where a virtual ball reacts to motion in a live feed.
- Compare the ergonomics of Python (fast prototyping) vs. C++ (fine-grained control, performance).

## Key Outcomes
- **Python version**: Feature tracking with trails; proof of concept.  
- **C++ version**: Localized ROI interaction; ball responds only to motion within its region.  
- **Cross-language learning**: Demonstrates equivalent logic in two ecosystems.

## Structure

optical_flow_ball/
    python/ #python implementation
    cpp/ #c++ implementation
    README.MD
