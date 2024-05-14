# Development of a Prototype Robot Automation Algorithm for Local Anesthesia

https://github.com/taeyeonmik/robotics_automation_algorithm/assets/161742505/138c8a9d-d529-4260-b41e-b77d54e9f422

## Project informations
- periode : 05/2023 - 07/2023
- purpose : to develop a prototype robot automation algorithm that will be applying in medical field, with an implementation of two deep learning detection models having different purposes.   
- lead by : Taeyeon Kim

## Project details
### 1. language and main frame works/libraries
- python
- pytorch, opencv-python, mediapipe, dobot

### 2. data
- image data of injured patients collected from the hospital. Due to the sensitivity of the data and requests from the company, it is not possible to disclose the data.

### 3. detection models
1) yolov7 
- purpose : detecting wounds or cut parts on hands.
2) hand landmarks keypoints detection model from Mediapipe
- purpose : obtaining all hand joint position information from the images.

### 4. Main Modules and Algorithms
1) main/train.py
   - training YOLOv7 
2) main/detect.py
   ![detect](https://github.com/taeyeonmik/robotics_automation_algorithm/assets/161742505/20e12f8e-f5c5-4b6b-91df-915cd35f9037)
   - detecting wounds and hand joints by using the trained YOLO and a pre-trained hand landmark keypoints detection model (Mideapipe) simultaneously.
3) main/injection.py
   ![injectionpoint](https://github.com/taeyeonmik/robotics_automation_algorithm/assets/161742505/a5fde0fc-1fa7-45e6-baca-04f3b776375e)
   - Development of an algorithm to find optimal local anesthesia points based on the locations of received wounds and joints according to medical criteria.
4) main/dobotcode/*.py
   - Management/Control of communication between robots and computers.
   - Control of the robot's position, direction of movement, speed, etc., based on received data.
5) algo.py
   - An integrated module that combines all the above-mentioned algorithm modules to execute detection and anesthesia in one step.
