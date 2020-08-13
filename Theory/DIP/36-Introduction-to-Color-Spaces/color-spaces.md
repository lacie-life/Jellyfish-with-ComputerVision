# Introduction to Color Spaces

## What are color spaces?
Color spaces are different types of color modes, used in image processing and signals and system for various purposes. Some of the common color spaces are:

+ RGB
+ CMY’K
+ Y’UV
+ YIQ
+ Y’CbCr
+ HSV

## RGB

![Figure 1](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/36-Introduction-to-Color-Spaces/colorspaces1.jpg?raw=true)

RGB is the most widely used color space, and we have already discussed it in the past tutorials. RGB stands for red green and blue.

What RGB model states, that each color image is actually formed of three different images. Red image, Blue image, and black image. A normal grayscale image can be defined by only one matrix, but a color image is actually composed of three different matrices.

One color image matrix = red matrix + blue matrix + green matrix

This can be best seen in this example below.

![Figure 2](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/36-Introduction-to-Color-Spaces/colorspaces2.jpg?raw=true)

### Applications of RGB
The common applications of RGB model are

+ Cathode ray tube (CRT)
+ Liquid crystal display (LCD)
+ Plasma Display or LED display such as a television
+ A compute monitor or a large scale screen

## CMYK

![Figure 3](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/36-Introduction-to-Color-Spaces/colorspaces3.jpg?raw=true)

### RGB to CMY conversion
The conversion from RGB to CMY is done using this method.

![Figure 4](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/36-Introduction-to-Color-Spaces/colorspaces4.jpg?raw=true)

Consider you have an color image , means you have three different arrays of RED, GREEN and BLUE. Now if you want to convert it into CMY, here’s what you have to do. You have to subtract it by the maximum number of levels – 1. Each matrix is subtracted and its respective CMY matrix is filled with result.

## Y'UV

Y’UV defines a color space in terms of one luma (Y’) and two chrominance (UV) components. The Y’UV color model is used in the following composite color video standards.

+ NTSC ( National Television System Committee)

+ PAL (Phase Alternating Line)

+ SECAM (Sequential couleur a amemoire, French for “sequential color with memory)

![Figure 5](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/36-Introduction-to-Color-Spaces/colorspaces5.jpg?raw=true)

## Y’CbCr
Y’CbCr color model contains Y’, the luma component and cb and cr are the blue-difference and red difference chroma components.

It is not an absolute color space. It is mainly used for digital systems

Its common applications include JPEG and MPEG compression.

Y’UV is often used as the term for Y’CbCr, however they are totally different formats. The main difference between these two is that the former is analog while the later is digital.

![Figure 6](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/36-Introduction-to-Color-Spaces/colorspaces6.jpg?raw=true)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------