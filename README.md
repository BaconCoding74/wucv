# Wuthering Wave Computer Vision (WUCV)
## Purpose
Since there is no official API from Wuthering Wave to extract items from inventory, my friend request me to make a tool to do so.

## Workflow
Capture frame from Wuthering Wave application 
-> Check pixel difference with previous frames 
-> If difference pass the threshold, perform item grid detection with some simple logic like canny edge detection, dilation, contour detection etc
-> If box founded, separating both quantity and item icon from the item grid
-> Perform quantity extraction with template matching (Number is the same, so OCR that cost more is excluded)
-> Perform item icon recognition with baseline DINOv2
-> Create new item if the item is new while update only quantity if item exist (Currently no DB exist as currently is still testing the capability to identity item)

## Current progression (Stagnant)
Currently this project is still in experimentation stage as my friend cannot provide me the dataset I want for feature testing and I don't plan to install Wuthering Wave also because of my limited storage space. However, currently the tool can work on Windows to capture the screen automatically and identify.

## Limitation / Challenges Right Now
- My friend seems not passionate about this project and take long time to provide assets I need for experimentation which cause the delaying of the development.
- I was busying with my own assignments and study at the middle of the project which make my friend loss their passionate.
- Some items share the same assets which make the item undistinguishable.
- Processing time is long (0.4 seconds) which means 30 FPS can only process 2 frames.
- This project currently only works with Window at experiment stage.
- My code and file structure is messy right now and I haven't organized it.

## What I learn
- Long development time will reduce the passionate of client (What being learn every time during university :P)
- Need to do more research when picking baseline model (Was choosing MobileNet that being trained with labelled dataset instead of DINOv2 that trained with supervised learning)
- Documentation is super super important, it helps me to recall what I did and which approach is more recommended with evidence.
- Use relative coordination is more stable when need to identify image at certain fixed location as different user have different resolution.
- Should git push more often.
- Create branch for experimentation so it wont makes file structure messy and allow me to revise back what approach I had been tested and its performance.
- Learn how to do image capturing, image processing and image retrieval with MSS, CV2 and PyTorch.
- Learn how to tackle issue background noise, wrong quantity extraction, wrong item grid detection etc.
