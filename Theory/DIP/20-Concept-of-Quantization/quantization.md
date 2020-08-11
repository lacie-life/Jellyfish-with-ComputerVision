# Concept of Quantization

We have introduced quantization in our tutorial of signals and system. We are formally going to relate it with digital images in this tutorial. Lets discuss first a little bit about quantization.

## Digitizing a signal

As we have seen in the previous tutorials, that digitizing an analog signal into a digital, requires two basic steps. Sampling and quantization. Sampling is done on x axis. It is the conversion of x axis (infinite values) to digital values.

The below figure shows sampling of a signal

![Figure 1](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/20-Concept-of-Quantization/sampling.jpg?raw=true)

### Sampling with relation to digital images
The concept of sampling is directly related to zooming. The more samples you take, the more pixels, you get. Oversampling can also be called as zooming. This has been discussed under sampling and zooming tutorial.

But the story of digitizing a signal does not end at sampling too, there is another step involved which is known as Quantization.

## What is Quantization

Quantization is opposite to sampling. It is done on y axis. When you are quantizing an image, you are actually dividing a signal into quanta(partitions).

On the x axis of the signal, are the co-ordinate values, and on the y axis, we have amplitudes. So digitizing the amplitudes is known as Quantization.

Here how it is done

![Figure 2](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/20-Concept-of-Quantization/quantization.jpg?raw=true)

You can see in this image, that the signal has been quantified into three different levels. That means that when we sample an image, we actually gather a lot of values, and in quantization, we set levels to these values. This can be more clear in the image below.

![Figure 3](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/20-Concept-of-Quantization/quantization_levels.jpg?raw=true)

In the figure shown in sampling, although the samples has been taken, but they were still spanning vertically to a continuous range of gray level values. In the figure shown above, these vertically ranging values have been quantized into 5 different levels or partitions. Ranging from 0 black to 4 white. This level could vary according to the type of image you want.

The relation of quantization with gray levels has been further discussed below.

Relation of Quantization with gray level resolution:

The quantized figure shown above has 5 different levels of gray. It means that the image formed from this signal, would only have 5 different colors. It would be a black and white image more or less with some colors of gray. Now if you were to make the quality of the image more better, there is one thing you can do here. Which is, to increase the levels, or gray level resolution up. If you increase this level to 256, it means you have an gray scale image. Which is far better then simple black and white image.

Now 256, or 5 or what ever level you choose is called gray level. Remember the formula that we discussed in the previous tutorial of gray level resolution which is,

![Figure 4](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/20-Concept-of-Quantization/bpp.jpg?raw=true)

## Reducing the gray level
Now we will reduce the gray levels of the image to see the effect on the image.

### For example
Lets say you have an image of 8bpp, that has 256 different levels. It is a grayscale image and the image looks something like this.

### 256 Gray Levels

![Figure 5](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/20-Concept-of-Quantization/einstein.jpg?raw=true)

Now we will start reducing the gray levels. We will first reduce the gray levels from 256 to 128.

### 128 Gray Levels

![Figure 6](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/20-Concept-of-Quantization/128.jpg?raw=true)

There is not much effect on an image after decrease the gray levels to its half. Lets decrease some more.

### 64 Gray Levels

![Figure 7](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/20-Concept-of-Quantization/64.jpg?raw=true)

Still not much effect, then lets reduce the levels more.

### 32 Gray Levels

![Figure 8](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/20-Concept-of-Quantization/32.jpg?raw=true)

Surprised to see, that there is still some little effect. May be its due to reason, that it is the picture of Einstein, but lets reduce the levels more.

### 16 Gray Levels

![Figure 8](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/20-Concept-of-Quantization/32.jpg?raw=true)

### 8 Gray Levels

![Figure 8](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/20-Concept-of-Quantization/8.jpg?raw=true)

### 4 Gray Levels

![Figure 8](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/20-Concept-of-Quantization/4.jpg?raw=true)

### 2 Gray Levels

![Figure 8](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/20-Concept-of-Quantization/2.jpg?raw=true)

## Contouring
There is an interesting observation here, that as we reduce the number of gray levels, there is a special type of effect start appearing in the image, which can be seen clear in 16 gray level picture. This effect is known as Contouring.

## Iso preference curves
The answer to this effect, that why it appears, lies in Iso preference curves. They are discussed in our next tutorial of Contouring and Iso preference curves.