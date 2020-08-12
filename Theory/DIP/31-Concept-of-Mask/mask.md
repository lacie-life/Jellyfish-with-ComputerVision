# Concept of Mask

## What is a mask
A mask is a filter. Concept of masking is also known as spatial filtering. Masking is also known as filtering. In this concept we just deal with the filtering operation that is performed directly on the image.

A sample mask has been shown below

![Figure 1](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/31-Concept-of-Mask/1.PNG?raw=true)

## What is flitering

The process of filtering is also known as convolving a mask with an image. As this process is same of convolution so filter masks are also known as convolution masks.

### How it is done

The general process of filtering and applying masks is consists of moving the filter mask from point to point in an image. At each point (x,y) of the original image, the response of a filter is calculated by a pre defined relationship. All the filters values are pre defined and are a standard.

### Types of fliters

Generally there are two types of filters. One is called as linear filters or smoothing filters and others are called as frequency domain filters.

### Why filters are used?
Filters are applied on image for multiple purposes. The two most common uses are as following:

+ Filters are used for Blurring and noise reduction
+ Filters are used or edge detection and sharpness

### Blurring and noise reduction
Filters are most commonly used for blurring and for noise reduction. Blurring is used in pre processing steps, such as removal of small details from an image prior to large object extraction.

### Masks for blurring
The common masks for blurring are.

+ Box filter
+ Weighted average filter
In the process of blurring we reduce the edge content in an image and try to make the transitions between different pixel intensities as smooth as possible.

Noise reduction is also possible with the help of blurring.

### Edge Detection and sharpness
Masks or filters can also be used for edge detection in an image and to increase sharpness of an image.

## What are edges

We can also say that sudden changes of discontinuities in an image are called as edges. Significant transitions in an image are called as edges.A picture with edges is shown below.

### Originel picture
![Figure 2](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/31-Concept-of-Mask/maskconcept1.jpg?raw=true)

### Same picture with edges
![Figure 3](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/31-Concept-of-Mask/maskconcept2.jpg?raw=true)

