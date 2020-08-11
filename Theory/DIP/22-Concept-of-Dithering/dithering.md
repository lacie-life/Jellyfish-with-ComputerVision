# Concept of Dithering (phối màu)

In the last two tutorials of Quantization and contouring, we have seen that reducing the gray level of an image reduces the number of colors required to denote an image. If the gray levels are reduced two 2, the image that appears doesnot have much spatial resolution or is not very much appealing.

## Dithering

Dithering is the process by which we create illusions of the color that are not present actually. It is done by the random arrangement of pixels.

For example. Consider this image.

![Figure 1](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/22-Concept-of-Dithering/dither1.jpg?raw=true)

This is an image with only black and white pixels in it. Its pixels are arranged in an order to form another image that is shown below. Note at the arrangement of pixels has been changed, but not the quantity of pixels.

![Figure 2](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/22-Concept-of-Dithering/dither2.jpg?raw=true)

### Why Dithering?
Why do we need dithering, the answer of this lies in its relation with quantization.

### Dithering with quantization
When we perform quantization, to the last level, we see that the image that comes in the last level (level 2) looks like this.

![Figure 3](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/22-Concept-of-Dithering/2.jpg?raw=true)

Now as we can see from the image here, that the picture is not very clear, especially if you will look at the left arm and back of the image of the Einstein. Also this picture does not have much information or detail of the Einstein.

Now if we were to change this image into some image that gives more detail then this, we have to perform dithering.

## Performing Dithering

First of all, we will work on threholding. Dithering is usually working to improve thresholding. During threholding, the sharp edges appear where gradients are smooth in an image.

In thresholding, we simply choose a constant value. All the pixels above that value are considered as 1 and all the value below it are considered as 0.

We got this image after thresholding.

![Figure 4](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/22-Concept-of-Dithering/dither3.jpg?raw=true)

Since there is not much change in the image, as the values are already 0 and 1 or black and white in this image.

Now we perform some random dithering to it. Its some random arrangement of pixels.

![Figure 5](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/22-Concept-of-Dithering/dither4.jpg?raw=true)

We got an image that gives slighter of the more details, but its contrast is very low.

So we do some more dithering that will increase the contrast. The image that we got is this:

![Figure 6](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/22-Concept-of-Dithering/dither5.jpg?raw=true)

Now we mix the concepts of random dithering, along with threshold and we got an image like this.

![Figure 7](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/22-Concept-of-Dithering/dither6.jpg?raw=true)

Now you see, we got all these images by just re-arranging the pixels of an image. This re-arranging could be random or could be according to some measure.