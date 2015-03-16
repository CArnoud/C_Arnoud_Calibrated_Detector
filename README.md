C_Arnoud's_Calibrated_Detector  
======================  
  
This is an attempt to improve the performance of pedestrian detectors using camera calibration.  
My C++ version of the Dóllar method is used for calculating the feature pyramid and classifying patches.    
Created by Charles Arnoud, under the mentorship of Cláudio Rosito Jüng and with the help of Gustavo Führ.  
  
  
Current Status  
======================  
  
Detection working, found out why the program was not optimized. 
Now it is faster than the original Matlab version, although results seem to be slightly worse.   
Calibrated detection and feature pyramid online, new candidate generation method is a work in progress.  
  
  
To Do List:  
======================  
  
Current Total: 3    
  
Calibrated Detection:  
&nbsp;&nbsp;&nbsp;&nbsp;Possibly add a way to use background information  
&nbsp;&nbsp;&nbsp;&nbsp;Remove magic numbers on calibrated detector and suppression  
&nbsp;&nbsp;&nbsp;&nbsp;debug function new candidate generation function  
      