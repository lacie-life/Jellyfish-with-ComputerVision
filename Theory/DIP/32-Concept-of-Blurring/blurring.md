# Concept of Blurring

## Blurring
In blurring, we simple blur an image. An image looks more sharp or more detailed if we are able to perceive all the objects and their shapes correctly in it. For example. An image with a face, looks clear when we are able to identify eyes, ears, nose, lips, forehead e.t.c very clear. This shape of an object is due to its edges. So in blurring, we simple reduce the edge content and makes the transition form one color to the other very smooth.

## Blurring vs zooming
You might have seen a blurred image when you zoom an image. When you zoom an image using pixel replication, and zooming factor is increased, you saw a blurred image. This image also has less details, but it is not true blurring.

Because in zooming, you add new pixels to an image, that increase the overall number of pixels in an image, whereas in blurring, the number of pixels of a normal image and a blurred image remains the same.

### Common exaple of a blurred image
![Figure 1](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/32-Concept-of-Blurring/blurring1.jpg?raw=true)

### Types of filters

Blurring can be achieved by many ways. The common type of filters that are used to perform blurring are.

+ Mean filter
+ Weighted average filter
+ Gaussian filter
Out of these three, we are going to discuss the first two here and Gaussian will be discussed later on in the upcoming tutorials.

## Mean filter
Mean filter is also known as Box filter and average filter. A mean filter has the following properties.

+ It must be odd ordered
+ The sum of all the elements should be 1
+ All the elements should be same
If we follow this rule, then for a mask of 3x3. We get the following result.

![Figure 2](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/32-Concept-of-Blurring/1.PNG?raw=true)

Since it is a 3x3 mask, that means it has 9 cells. The condition that all the element sum should be equal to 1 can be achieved by dividing each value by 9. As

1/9 + 1/9 + 1/9 + 1/9 + 1/9 + 1/9 + 1/9 + 1/9 + 1/9 = 9/9 = 1

### The result of a mask of 3x3 on an image is shown below

|Original Image|Blurred Image|
|--------------|--------------|
|![Figure 3](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/32-Concept-of-Blurring/blurring2.jpg?raw=true)|![Figure 4](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/32-Concept-of-Blurring/blurring3.jpg?raw=true)|

May be the results are not much clear. Let’s increase the blurring. The blurring can be increased by increasing the size of the mask. The more is the size of the mask, the more is the blurring. Because with greater mask, greater number of pixels are catered and one smooth transition is defined.

### The result of a mask of 5x5 on an image is shown below

|Original Image|Blurred Image|
|--------------|--------------|
|![Figure 5](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/32-Concept-of-Blurring/blurring4.jpg?raw=true)|![Figure 6](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/32-Concept-of-Blurring/blurring5.jpg?raw=true)|


Same way if we increase the mask, the blurring would be more and the results are shown below.

### The result of a mask of 7x7 on an image is shown below.

|Original Image|Blurred Image|
|--------------|--------------|
|![Figure 7](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/32-Concept-of-Blurring/blurring6.jpg?raw=true)|![Figure 8](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/32-Concept-of-Blurring/blurring7.jpg?raw=true)|

### The result of a mask of 9x9 on an image is shown below.

|Original Image|Blurred Image|
|--------------|--------------|
|![Figure 9](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/32-Concept-of-Blurring/blurring8.jpg?raw=true)|![Figure 10](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/32-Concept-of-Blurring/blurring9.jpg?raw=true)|

### The result of a mask of 11x11 on an image is shown below.

|Original Image|Blurred Image|
|--------------|--------------|
|![Figure 11](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/32-Concept-of-Blurring/blurring10.jpg?raw=true)|![Figure 12](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/32-Concept-of-Blurring/blurring10.jpg?raw=true)|

## Weighted average filter
In weighted average filter, we gave more weight to the center value. Due to which the contribution of center becomes more then the rest of the values. Due to weighted average filtering, we can actually control the blurring.

Properties of the weighted average filter are.

+ It must be odd ordered
+ The sum of all the elements should be 1
+ The weight of center element should be more then all of the other elements

![Figure 13](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/32-Concept-of-Blurring/2.PNG?raw=true)