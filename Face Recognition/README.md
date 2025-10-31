# Table of Contents
- [Face recognition revision](#face-recognition-revision)
- [Issue](#issue)
- [What I added](#what-I-added)
- [Limitations](#limitations)




## Face recognition revision

- An addition to [Core Electronics -> "Facial Recognition for Raspberry Pi with OpenCV and Python (Updated Tutorial)"](https://www.youtube.com/watch?v=3TUlJrRJUeM)

- It is best to click on this link to fully understand what Core Electronics created an how it works to see what changes I made
    
- Here is the link to the original [code](https://core-electronics.com.au/guides/raspberry-pi/face-recognition-with-raspberry-pi-and-opencv/)
    - It is a webpage and you can scroll down to find the link and upload the source files



## Issue

- As highlighted by Jaryd, the application can has frame and detection issues. 

    --- The higher the cv_scaler, the less accurate it gets in detecting, but the frames are abit faster

    --- The lower the cv_scaler, the more accurate it gets in detecting. However the slower the frames
    
## What I Added

 - To make a meaningful increase in fps while detecting while keeping a decent amount of accuracy, I used the python [threading](https://docs.python.org/3/library/threading.html) library 

- By implementing two queues (frame_queue & result_queue ) & threading library

    -- the function face_rec() can concurrently call frame_queue. 

    -- Find the faces and distances.

    -- Then return the frame to result_queue



    -- In main, each incoming frame gets sent to frame_queue. 

    --The next frame in result_queue gets grabbed and input bounding boxes & names. 

    -- Then output the frames to the user

## Limitations

 - you can adjust the "cv_scaler" by changing the directions numbers; So top *= 2, bottom *= 2
        ```bash
            
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2


   

- However, the bounding boxes can shift abit wildy compared to original source.

- you can increase the fps by adjusting
        ``` bash

            if frame_count % 5 == 0

    
_Note the smaller the number the more "blinky" the bounding boxes become. But increasingly accurate._

_The higher the number, the more steady but less location accuracy of the boxes if you move around_