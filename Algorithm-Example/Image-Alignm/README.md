## Homography

Two images of a scense are realted by a homography under two conditions.

1. The two images that of a plane 
2. The two images were accquired by rotating the camera about its optical axis. We take such images whilt generating panoramas

As metions earlier, a homography is nothing but a 3x3 matrix as shown below:

![Homography 1](https://github.com/lacie-life/Imagpe-Processing/blob/master/Example/Image-Alignm/images/matrix.png?raw=true)

Let (x1, y1) be a point in the first image and (x2, y2) be the coordinates of the same physical point in the second image. Then, the Homography H relates them in the following way:

![Homography 2](https://github.com/lacie-life/Imagpe-Processing/blob/master/Example/Image-Alignm/images/homography.png?raw=true)

If we knew the homography, we could apply it to all the pixels of one image to obtain a warped image that is aligned with the second image.

[Refer](https://learnopencv.com/feature-based-image-alignment-using-opencv-c-python/)

- Read Images
- Detect Features
- Match Features
- Calculate Homography (RANSAC)
- Warping image