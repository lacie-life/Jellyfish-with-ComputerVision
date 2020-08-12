# Histogram Sliding

### Histogram
Histogram is nothing but a graph that shows frequency of occurrence of data. Histograms has many use in image processing, out of which we are going to discuss one user here which is called histogram sliding.


### Histogram sliding
In histogram sliding, we just simply shift a complete histogram rightwards or leftwards. Due to shifting or sliding of histogram towards right or left, a clear change can be seen in the image.In this tutorial we are going to use histogram sliding for manipulating brightness.

The term i-e: Brightness has been discussed in our tutorial of introduction to brightness and contrast. But we are going to briefly define here.

### Brightness
Brightness is a relative term. Brightness can be defined as intensity of light emit by a particular light source.

### Contrast
Contrast can be defined as the difference between maximum and minimum pixel intensity in an image.

## Sliding Histograms

Increasing brightness using histogram sliding

![Figure 1](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/26-Histogram/einstein.jpg?raw=true)

Histogram of this image has been shown below.

![Figure 2](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/26-Histogram/histogram1.jpg?raw=true)

On the y axis of this histogram are the frequency or count. And on the x axis, we have gray level values. As you can see from the above histogram, that those gray level intensities whose count is more then 700, lies in the first half portion, means towards blacker portion. Thats why we got an image that is a bit darker.

In order to bright it, we will slide its histogram towards right, or towards whiter portion. In order to do we need to add atleast a value of 50 to this image. Because we can see from the histogram above, that this image also has 0 pixel intensities, that are pure black. So if we add 0 to 50, we will shift all the values lies at 0 intensity to 50 intensity and all the rest of the values will be shifted accordingly.

Lets do it.

### Here what we got after adding 50 to each pixel intensity.

The image has been shown below.

![Figure 3](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/26-Histogram/sliding2.jpg?raw=true)

And its histogram has been shown below.

![Figure 4](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/26-Histogram/sliding3.jpg?raw=true)

Lets compare these two images and their histograms to see that what change have to got.

![Figure 5](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/26-Histogram/sliding4.jpg?raw=true)

### Conclusion
As we can clearly see from the new histogram that all the pixels values has been shifted towards right and its effect can be seen in the new image.

## Decreasing brightness using histogram sliding

Now if we were to decrease brightness of this new image to such an extent that the old image look brighter, we got to subtract some value from all the matrix of the new image. The value which we are going to subtract is 80. Because we already add 50 to the original image and we got a new brighter image, now if we want to make it darker, we have to subtract at least more than 50 from it.

And this what we got after subtracting 80 from the new image.

![Figure 6](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/26-Histogram/sliding5.jpg?raw=true)

### Conclusion
It is clear from the histogram of the new image, that all the pixel values has been shifted towards right and thus, it can be validated from the image that new image is darker and now the original image look brighter as compare to this new image.

------------------------------------------------------------------------------------

# Histogram stretching

One of the other advantage of Histogram s that we discussed in our tutorial of introduction to histograms is contrast enhancement.

There are two methods of enhancing contrast. The first one is called Histogram stretching that increase contrast. The second one is called Histogram equalization that enhance contrast and it has been discussed in our tutorial of histogram equalization.

Before we will discuss the histogram stretching to increase contrast, we will briefly define contrast.

## Contrast

Contrast is the difference between maximum and minimum pixel intensity.

Consider this image.

![Figure 7](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/26-Histogram/stretching1.jpg?raw=true)

The histogram of this image is shown below.

![Figure 7](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/26-Histogram/stretching2.jpg?raw=true)

Now we calculate contrast from this image.

Contrast = 225.

Now we will increase the contrast of the image.

## Increasing the contrast of the image
The formula for stretching the histogram of the image to increase the contrast is

![Figure 8](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/26-Histogram/stretching3.jpg?raw=true)

The formula requires finding the minimum and maximum pixel intensity multiply by levels of gray. In our case the image is 8bpp, so levels of gray are 256.

The minimum value is 0 and the maximum value is 225. So the formula in our case is

![Figure 9](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/26-Histogram/stretching4.jpg?raw=true)

where f(x,y) denotes the value of each pixel intensity. For each f(x,y) in an image , we will calculate this formula.

After doing this, we will be able to enhance our contrast.

The following image appear after applying histogram stretching.

![Figure 10](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/26-Histogram/stretching5.jpg?raw=true)

The stretched histogram of this image has been shown below.

Note the shape and symmetry of histogram. The histogram is now stretched or in other means expand. Have a look at it.

![Figure 11](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/26-Histogram/stretching6.jpg?raw=true)

In this case the contrast of the image can be calculated as

Contrast = 240

Hence we can say that the contrast of the image is increased.

Note : this method of increasing contrast doesnot work always, but it fails on some cases.

## Failing of histogram stretching

As we have discussed , that the algorithm fails on some cases. Those cases include images with when there is pixel intensity 0 and 255 are present in the image

Because when pixel intensities 0 and 255 are present in an image, then in that case they become the minimum and maximum pixel intensity which ruins the formula like this.

Original Formula

![Figure 12](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/26-Histogram/stretching3.jpg?raw=true)

Putting fail case values in the formula:

![Figure 13](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/26-Histogram/stretching7.jpg?raw=true)

Simplify that expression gives

![Figure 14](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/26-Histogram/stretching8.jpg?raw=true)

That means the output image is equal to the processed image. That means there is no effect of histogram stretching has been done at this image.