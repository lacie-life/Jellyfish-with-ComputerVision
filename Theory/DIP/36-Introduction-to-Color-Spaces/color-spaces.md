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

# Introduction to JPEG Compression

In our last tutorial of image compression, we discuss some of the techniques used for compression

We are going to discuss JPEG compression which is lossy compression, as some data is loss in the end.

Let’s discuss first what image compression is.

## Image compression
Image compression is the method of data compression on digital images.

The main objective in the image compression is:

+ Store data in an efficient form
+ Transmit data in an efficient form
Image compression can be lossy or lossless.

## JPEG compression
JPEG stands for Joint photographic experts group. It is the first interanational standard in image compression. It is widely used today. It could be lossy as well as lossless . But the technique we are going to discuss here today is lossy compression technique.

### How jpeg compression works
First step is to divide an image into blocks with each having dimensions of 8 x8.

![Figure 7](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/36-Introduction-to-Color-Spaces/compression1.jpg?raw=true)

Let’s for the record, say that this 8x8 image contains the following values.

![Figure 8](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/36-Introduction-to-Color-Spaces/compression2.jpg?raw=true)

The range of the pixels intensities now are from 0 to 255. We will change the range from -128 to 127.

Subtracting 128 from each pixel value yields pixel value from -128 to 127. After subtracting 128 from each of the pixel value, we got the following results.

![Figure 9](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/36-Introduction-to-Color-Spaces/compression3.jpg?raw=true)

Now we will compute using this formula.

![Figure 10](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/36-Introduction-to-Color-Spaces/compression4.jpg?raw=true)

The result comes from this is stored in let’s say A(j,k) matrix.

There is a standard matrix that is used for computing JPEG compression, which is given by a matrix called as Luminance matrix.

This matrix is given below

![Figure 11](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/36-Introduction-to-Color-Spaces/compression5.jpg?raw=true)

Applying the following formula

![Figure 12](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/36-Introduction-to-Color-Spaces/compression6.jpg?raw=true)

We got this result after applying.

![Figure 13](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/36-Introduction-to-Color-Spaces/compression7.jpg?raw=true)

Now we will perform the real trick which is done in JPEG compression which is ZIG-ZAG movement. The zig zag sequence for the above matrix is shown below. You have to perform zig zag until you find all zeroes ahead. Hence our image is now compressed.

![Figure 14](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/36-Introduction-to-Color-Spaces/compression7.jpg?raw=true)

### Summarizing JPEG compression
The first step is to convert an image to Y’CbCr and just pick the Y’ channel and break into 8 x 8 blocks. Then starting from the first block, map the range from -128 to 127. After that you have to find the discrete Fourier transform of the matrix. The result of this should be quantized. The last step is to apply encoding in the zig zag manner and do it till you find all zero.

Save this one dimensional array and you are done.

Note. You have to repeat this procedure for all the block of 8 x 8.