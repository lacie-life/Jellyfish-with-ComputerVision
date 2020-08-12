# Concept of Convolution

## What is image processing

As we have discussed in the introduction to image processing tutorials and in the signal and system that image processing is more or less the study of signals and systems because an image is nothing but a two dimensional signal.

Also we have discussed, that in image processing , we are developing a system whose input is an image and output would be an image. This is pictorially represented as.

![Figure 1](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/30-Concept-of-Convolution/conceptconvolution1.jpg?raw=true)

The box is that is shown in the above figure labeled as “Digital Image Processing system” could be thought of as a black box

It can be better represented as:

![Figure 2](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/30-Concept-of-Convolution/conceptconvolution2.jpg?raw=true)

## Where have we reached until now

Till now we have discussed two important methods to manipulate images. Or in other words we can say that, our black box works in two different ways till now.

The two different ways of manipulating images were

### Graphs (Histograms)

![Figure 3](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/30-Concept-of-Convolution/conceptconvolution3.jpg?raw=true)

This method is known as histogram processing. We have discussed it in detail in previous tutorials for increase contrast, image enhancement, brightness e.t.c

### Transformation functions

![Figure 4](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/30-Concept-of-Convolution/conceptconvolution4.jpg?raw=true)

This method is known as transformations, in which we discussed different type of transformations and some gray level transformations

### Another way of dealing images

Here we are going to discuss another method of dealing with images. This other method is known as convolution. Usually the black box(system) used for image processing is an LTI system or linear time invariant system. By linear we mean that such a system where output is always linear , neither log nor exponent or any other. And by time invariant we means that a system which remains same during time.

So now we are going to use this third method. It can be represented as

![Figure 5](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/30-Concept-of-Convolution/conceptconvolution5.jpg?raw=true)

It can be mathematically represented as two ways

g(x,y) = h(x,y) * f(x,y)

It can be explained as the “mask convolved with an image”.

Or

g(x,y) = f(x,y) * h(x,y)

It can be explained as “image convolved with mask”.

There are two ways to represent this because the convolution operator(*) is commutative. The h(x,y) is the mask or filter.

## What is mask?

Mask is also a signal. It can be represented by a two dimensional matrix. The mask is usually of the order of 1x1, 3x3, 5x5, 7x7 . A mask should always be in odd number, because other wise you cannot find the mid of the mask. Why do we need to find the mid of the mask. The answer lies below, in topic of, how to perform convolution?

## How to perform convolution?

In order to perform convolution on an image, following steps should be taken.

+ Flip the mask (horizontally and vertically) only once
+ Slide the mask onto the image.
+ Multiply the corresponding elements and then add them
+ Repeat this procedure until all values of the image has been calculated.

## Example of convolution

Let’s perform some convolution. Step 1 is to flip the mask.

![Figure 6](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/30-Concept-of-Convolution/1.PNG?raw=true)

## Convolution

Convolving mask over image. It is done in this way. Place the center of the mask at each element of an image. Multiply the corresponding elements and then add them , and paste the result onto the element of the image on which you place the center of mask.

![Figure 7](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/30-Concept-of-Convolution/conceptconvolution6.jpg?raw=true)

The box in red color is the mask, and the values in the orange are the values of the mask. The black color box and values belong to the image. Now for the first pixel of the image, the value will be calculated as

First pixel = (5*2) + (4*4) + (2*8) + (1*10)

= 10 + 16 + 16 + 10

= 52

Place 52 in the original image at the first index and repeat this procedure for each pixel of the image.

## Why Convolution

Convolution can achieve something, that the previous two methods of manipulating images can’t achieve. Those include the blurring, sharpening, edge detection, noise reduction e.t.c.
