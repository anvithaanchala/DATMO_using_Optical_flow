# DATMO_using_Optical_flow


Names: Venkata Satya Naga Sai Karthik kodur
System Info 

Project Description

This project focuses on evaluating and comparing the performance of two advanced algorithms, Optical Flow and the General Model-Free Approach (GMFA), for the Detection and Tracking of Moving Objects (DATMO) in autonomous vehicles. The study leverages the CARLA simulation environment to generate realistic and complex traffic scenarios, enabling robust testing of these methodologies under diverse conditions.

Optical Flow uses pixel motion in sequential frames to estimate velocities, incorporating preprocessing techniques such as RANSAC-based ground segmentation and continuity masks to reduce false positives. GMFA, on the other hand, relies on geometric model independence, classifying objects based on residual motion analysis without predefined shapes, making it adaptable to heterogeneous environments.

Key evaluation metrics include velocity estimation accuracy, detection precision, recall, and computational efficiency. The results highlight each algorithm's strengths and weaknesses, offering insights into their suitability for real-world applications. By critically analyzing their performance, this research aims to advance DATMO systems, addressing challenges like dynamic environments and sparse data. Future work will explore adaptive parameter tuning, diverse environmental conditions, and sensor fusion to enhance robustness and scalability. The findings contribute to developing efficient perception systems for autonomous driving technologies.


How to Run single_target_simulation.py Script

We used Carla 9.12 for sim

First Install the requirements 
Requirements 

Python 3.8 or later
glob
os
sys
random
numpy
datetime
argparse
open3d
cv2
math

How to run Optical Flow files :
1.Go to Optical_flow folder 
2.Change the parameters according to your dataset 
3.Run main.py 
4.Use saving_utils.py in case you want to save your bev , velocity vector grids , DBSCAN clusters and EKF tracked outputs

How to run GMFA code  files 