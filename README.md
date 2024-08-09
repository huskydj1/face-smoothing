# Face Smoothing: Detection and Beautification

## Adapted from [5starkarma's Original Repository](https://github.com/5starkarma/face-smoothing)

OpenCV implementation of facial smoothing. 

General Pipeline:

- Face Detection
- Filtering
        - Change image from BGR to HSV colorspace
        - Create mask of HSV image
        - Apply a bilateral filter to the Region of Interest
        - Apply filtered ROI back to original image
- Output

Areas for Improvement:

- Face segmentation to only blur skin regions
- Different Filtering Schemas (not just smoothing, contouring too)

## Run

```

python3 infer.py --input 'path/to/input_file.jpg' (Input file - image, video, or folder with images and/or videos - default is hillary_clinton.jpg)
                         'can/handle/videos.mp4'
                         'as/well/as/directories'
                 --output 'path/to/output_folder' (Output folder - default is data/output)
                 --save_steps 'path/to/file.jpg' (Concats images from each step of the process and saves them)
                 --show-detections (Saves bounding box detections to output)
```
#### Example: --save-steps flag
![alt text](https://github.com/5starkarma/face-smoothing/blob/main/data/output/combined_0.jpg?raw=true "Processing steps")


<details>
        <summary>Work completed by 5starkarma</summary>
- [X] Finish documentation and cleanup functions
- [X] Reduce input image size for detections
- [X] Fix combined output
- [X] Test on multiple faces
- [X] Apply blurring on multiple faces
- [X] Video inference
- [X] Save bounding box to output
- [ ] Apply different blurring techniques/advanced algo using facial landmarks to blur only skin regions
- [ ] Unit tests
- [ ] Run time tests on units

</details>
