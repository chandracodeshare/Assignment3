## Assignment 3: Vision-based control of a robotic arm

## Overview:  
This assignment focuses on implementing and analyzing vision-based control strategies for robotic manipulators.

## Simulation Environment  

The simulation environment is PyBullet, which provides realistic kinematics and dynamic utilities for a robotic along with real-time camera images. 
<p align="center">
  <img src="env.jpg" alt="Simulation Environment" width="600"/>
</p>

### Exercises  
The assignment has 6 exercises:

* **Part 1**: Eye-in-hand object position estimation
* **Part 2**: Eye-to-hand object position estimation
* **Part 3**: Image-based visual-servoing for reaching the static object
* **Part 4**: Image-based visual-servoing for tracking the moving object
	
For **Part 1 and Part 2**, estimate the position (x, y, z) of a target object using the provided camera setup (end-effector-mounted for eye-in-hand, fixed external camera for eye-to-hand). Display the estimated object position and true position in the environment or terminal.

For **Part 3**, the goal is to control the robot end-effector so that it reaches a desired pose relative to a static object. You should also plot the error in the image space (pixel error vs time).

For **Part 4**, extend your IBVS implementation to track a moving object.

For **all parts**, you are expected to deliver the code implementation

#### Extra Credits
* Implement occlusion handling using robust estimation or filtering.
 
## Setup Instructions  

### Requirements  
- Python 3.8+  
- [PyBullet](https://pybullet.org): Install using `pip install pybullet`
- NumPy 
- OpenCV: Install using `pip install opencv-python` and `pip install opencv-contrib-python`
