# Convolution Theorem

In the last tutorial, we discussed about the images in frequency domain. In this tutorial, we are going to define a relationship between frequency domain and the images(spatial domain).

### For example

Consider this example.

![Figure 1](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/35-Convolution-Theorem/convolution1.jpg?raw=true)

The same image in the frequency domain can be represented as.

![Figure 2](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/35-Convolution-Theorem/convolution2.jpg?raw=true)

Now what’s the relationship between image or spatial domain and frequency domain. This relationship can be explained by a theorem which is called as Convolution theorem.

## Convolution Theorem
The relationship between the spatial domain and the frequency domain can be established by convolution theorem.

The convolution theorem can be represented as.

![Figure 3](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/35-Convolution-Theorem/convolution3.jpg?raw=true)

It can be stated as the convolution in spatial domain is equal to filtering in frequency domain and vice versa.

The filtering in frequency domain can be represented as following:

![Figure 4](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/35-Convolution-Theorem/convolution4.jpg?raw=true)

### The steps in filtering are given below.

+ At first step we have to do some pre – processing an image in spatial domain, means increase its contrast or brightness

+ Then we will take discrete Fourier transform of the image

+ Then we will center the discrete Fourier transform, as we will bring the discrete Fourier transform in center from corners

+ Then we will apply filtering, means we will multiply the Fourier transform by a filter function

+ Then we will again shift the DFT from center to the corners

+ Last step would be take to inverse discrete Fourier transform, to bring the result back from frequency domain to spatial domain

+ And this step of post processing is optional, just like pre processing , in which we just increase the appearance of image.


## Filters
The concept of filter in frequency domain is same as the concept of a mask in convolution.

After converting an image to frequency domain, some filters are applied in filtering process to perform different kind of processing on an image. The processing include blurring an image, sharpening an image e.t.c.

The common type of filters for these purposes are:

+ Ideal high pass filter
+ Ideal low pass filter
+ Gaussian high pass filter
+ Gaussian low pass filter
In the next tutorial, we will discuss about filter in detail.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

# High Pass vs Low Pass Filters

In the last tutorial, we briefly discuss about filters. In this tutorial we will thoroughly discuss about them. Before discussing about let’s talk about masks first. The concept of mask has been discussed in our tutorial of convolution and masks.

## Blurring masks vs derivative masks

We are going to perform a comparison between blurring masks and derivative masks.

### Blurring masks
A blurring mask has the following properties.

+ All the values in blurring masks are positive
+ The sum of all the values is equal to 1
+ The edge content is reduced by using a blurring mask
+ As the size of the mask grow, more smoothing effect will take place

### Derivative masks
A derivative mask has the following properties.

+ A derivative mask have positive and as well as negative values
+ The sum of all the values in a derivative mask is equal to zero
+ The edge content is increased by a derivative mask
+ As the size of the mask grows , more edge content is increased

### Relationship between blurring mask and derivative mask with high pass filters and low pass filters.
The relationship between blurring mask and derivative mask with a high pass filter and low pass filter can be defined simply as.

+ Blurring masks are also called as low pass filter
+ Derivative masks are also called as high pass filter

### High pass frequency components and Low pass frequency components
The high pass frequency components denotes edges whereas the low pass frequency components denotes smooth regions.

### Ideal low pass and Ideal High pass filters
This is the common example of low pass filter.

![Figure 5](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/35-Convolution-Theorem/highpass1.jpg?raw=true)

When one is placed inside and the zero is placed outside , we got a blurred image. Now as we increase the size of 1, blurring would be increased and the edge content would be reduced.

This is a common example of high pass filter.

![Figure 6](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/35-Convolution-Theorem/highpass2.jpg?raw=true)

When 0 is placed inside, we get edges, which gives us a sketched image. An ideal low pass filter in frequency domain is given below.

![Figure 7](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/35-Convolution-Theorem/highpass3.jpg?raw=true)

The ideal low pass filter can be graphically represented as

![Figure 8](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/35-Convolution-Theorem/highpass4.jpg?raw=true)

Now let’s apply this filter to an actual image and let’s see what we got.

### Sample Image
![Figure 9](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/35-Convolution-Theorem/highpass5.jpg?raw=true)

### Image in frequency domain
![Figure 10](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/35-Convolution-Theorem/highpass6.jpg?raw=true)

### Applying filter over this image
![Figure 11](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/35-Convolution-Theorem/highpass7.jpg?raw=true)

### Resultant Image
![Figure 12](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/35-Convolution-Theorem/highpass8.jpg?raw=true)

With the same way, an ideal high pass filter can be applied on an image. But obviously the results would be different as, the low pass reduces the edged content and the high pass increase it.

## Gaussian Low pass and Gaussian High pass filter
Gaussian low pass and Gaussian high pass filter minimize the problem that occur in ideal low pass and high pass filter.

This problem is known as ringing effect. This is due to reason because at some points transition between one color to the other cannot be defined precisely, due to which the ringing effect appears at that point.

Have a look at this graph.

![Figure 13](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/35-Convolution-Theorem/highpass9.jpg?raw=true)

This is the representation of ideal low pass filter. Now at the exact point of Do, you cannot tell that the value would be 0 or 1. Due to which the ringing effect appears at that point.

So in order to reduce the effect that appears is ideal low pass and ideal high pass filter, the following Gaussian low pass filter and Gaussian high pass filter is introduced.

### Gaussian Low pass filter
The concept of filtering and low pass remains the same, but only the transition becomes different and become more smooth.

The Gaussian low pass filter can be represented as

![Figure 14](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/35-Convolution-Theorem/highpass10.jpg?raw=true)

Note the smooth curve transition, due to which at each point, the value of Do, can be exactly defined.

### Gaussian high pass filter
Gaussian high pass filter has the same concept as ideal high pass filter, but again the transition is more smooth as compared to the ideal one.
