C_Arnoud's_Calibrated_Detector  
======================  
  
This is an attempt to improve the performance of pedestrian detectors using camera calibration.  
My C++ version of the Dóllar method is used for calculating the feature pyramid and classifying patches.    
Created by Charles Arnoud, under the mentorship of Cláudio Rosito Jüng and with the help of Gustavo Führ.  
  
  
Current Status  
======================  
  
Detection working, but is slower than in the original version (implementation problem).   
Calibrated detection and feature pyramid online, new candidate generation method under way.  
  
  
To Do List:  
======================  
  
Current Total: 4    
  
C_Arnoud_Calibrated_Detector.cpp:  
&nbsp;&nbsp;&nbsp;&nbsp;find out why the program is so slow!  
  
Calibrated Detection:  
&nbsp;&nbsp;&nbsp;&nbsp;Possibly add a way to use background information  
&nbsp;&nbsp;&nbsp;&nbsp;Remove magic numbers on calibrated detector and suppression  
&nbsp;&nbsp;&nbsp;&nbsp;Create a new getCandidates function   
      