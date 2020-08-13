# Introduction to Frequency domain

We have deal with images in many domains. Now we are processing signals (images) in frequency domain. Since this Fourier series and frequency domain is purely mathematics, so we will try to minimize that math’s part and focus more on its use in DIP.

## Frequency domain analysis
Till now, all the domains in which we have analyzed a signal , we analyze it with respect to time. But in frequency domain we don’t analyze signal with respect to time, but with respect of frequency.

### Difference between spatial domain and frequency domain
In spatial domain, we deal with images as it is. The value of the pixels of the image change with respect to scene. Whereas in frequency domain, we deal with the rate at which the pixel values are changing in spatial domain.

For simplicity, Let’s put it this way.

### Spatial domain

![Figure 1](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/34-Introduction-to-Frequency-domain/frequencydomain1.jpg?raw=true)

In simple spatial domain, we directly deal with the image matrix. Whereas in frequency domain, we deal an image like this.

### Frequency Domain
We first transform the image to its frequency distribution. Then our black box system perform what ever processing it has to performed, and the output of the black box in this case is not an image, but a transformation. After performing inverse transformation, it is converted into an image which is then viewed in spatial domain.

It can be pictorially viewed as

![Figure 2](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/34-Introduction-to-Frequency-domain/frequencydomain2.jpg?raw=true)

### Transformation
A signal can be converted from time domain into frequency domain using mathematical operators called transforms. There are many kind of transformation that does this. Some of them are given below.

+ Fourier Series
+ Fourier transformation
+ Laplace transform
+ Z transform
Out of all these, we will thoroughly discuss Fourier series and Fourier transformation in our next tutorial.

## Frequency components
Any image in spatial domain can be represented in a frequency domain. But what do this frequencies actually mean.

We will divide frequency components into two major components.

### High frequency components
High frequency components correspond to edges in an image.

### Low frequency components
Low frequency components in an image correspond to smooth regions.

----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Fourier Series and Transform

In the last tutorial of Frequency domain analysis, we discussed that Fourier series and Fourier transform are used to convert a signal to frequency domain.

## Fourier

Fourier was a mathematician in 1822. He give Fourier series and Fourier transform to convert a signal into frequency domain.

### Fourier Series
Fourier series simply states that, periodic signals can be represented into sum of sines and cosines when multiplied with a certain weight.It further states that periodic signals can be broken down into further signals with the following properties.

+ The signals are sines and cosines
+ The signals are harmonics of each other
It can be pictorially viewed as

![Figure 3](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/34-Introduction-to-Frequency-domain/fourier1.jpg?raw=true)

In the above signal, the last signal is actually the sum of all the above signals. This was the idea of the Fourier.

### How it is calculated
Since as we have seen in the frequency domain, that in order to process an image in frequency domain, we need to first convert it using into frequency domain and we have to take inverse of the output to convert it back into spatial domain. That’s why both Fourier series and Fourier transform has two formulas. One for conversion and one converting it back to the spatial domain.

### Fourier series
The Fourier series can be denoted by this formula.

![Figure 4](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/34-Introduction-to-Frequency-domain/fourier2.jpg?raw=true)

The inverse can be calculated by this formula.

![Figure 5](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/34-Introduction-to-Frequency-domain/fourier3.jpg?raw=true)

## Fourier transform
The Fourier transform simply states that that the non periodic signals whose area under the curve is finite can also be represented into integrals of the sines and cosines after being multiplied by a certain weight.

The Fourier transform has many wide applications that include, image compression (e.g JPEG compression), filtering and image analysis.

## Difference between Fourier series and transform

Although both Fourier series and Fourier transform are given by Fourier , but the difference between them is Fourier series is applied on periodic signals and Fourier transform is applied for non periodic signals

### Which one is applied on images
Now the question is that which one is applied on the images , the Fourier series or the Fourier transform. Well, the answer to this question lies in the fact that what images are. Images are non – periodic. And since the images are non periodic, so Fourier transform is used to convert them into frequency domain.

### Discrete fourier transform
Since we are dealing with images, and in fact digital images, so for digital images we will be working on discrete fourier transform

![Figure 6](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/34-Introduction-to-Frequency-domain/fourier4.jpg?raw=true)

Consider the above Fourier term of a sinusoid. It include three things.

+ Spatial Frequency
+ Magnitude
+ Phase

The spatial frequency directly relates with the brightness of the image. The magnitude of the sinusoid directly relates with the contrast. Contrast is the difference between maximum and minimum pixel intensity. Phase contains the color information.

The formula for 2 dimensional discrete Fourier transform is given below.

![Figure 7](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/34-Introduction-to-Frequency-domain/fourier5.jpg?raw=true)

The discrete Fourier transform is actually the sampled Fourier transform, so it contains some samples that denotes an image. In the above formula f(x,y) denotes the image, and F(u,v) denotes the discrete Fourier transform. The formula for 2 dimensional inverse discrete Fourier transform is given below.

![Figure 8](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/34-Introduction-to-Frequency-domain/fourier6.jpg?raw=true)

The inverse discrete Fourier transform converts the Fourier transform back to the image

### Consider this signal
Now we will see an image, whose we will calculate FFT magnitude spectrum and then shifted FFT magnitude spectrum and then we will take Log of that shifted spectrum.

### Original Image

![Figure 9](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/34-Introduction-to-Frequency-domain/fourier7.jpg?raw=true)

### The Fourier transform magnitude spectrum

![Figure 10](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/34-Introduction-to-Frequency-domain/fourier8.jpg?raw=true)

### The Shifted Fourier transform

![Figure 11](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/34-Introduction-to-Frequency-domain/fourier9.jpg?raw=true)

### The Shifted Magnitude Spectrum

![Figure 12](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/34-Introduction-to-Frequency-domain/fourier10.jpg?raw=true)
