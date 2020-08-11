# ISO preference curves

## What is contouring?

As we decrease the number of gray levels in an image, some false colors, or edges start appearing on an image. This has been shown in our last tutorial of Quantization.

Lets have a look at it.

Consider we, have an image of 8bpp (a grayscale image) with 256 different shades of gray or gray levels.

![Figure 1](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/21-ISO-perference-curves/einstein.jpg?raw=true)

This above picture has 256 different shades of gray. Now when we reduce it to 128 and further reduce it 64, the image is more or less the same. But when re reduce it further to 32 different levels, we got a picture like this

![Figure 2](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/21-ISO-perference-curves/32.jpg?raw=true)

If you will look closely, you will find that the effects start appearing on the image.These effects are more visible when we reduce it further to 16 levels and we got an image like this.

![Figure 3](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/21-ISO-perference-curves/16.jpg?raw=true)

These lines, that start appearing on this image are known as contouring that are very much visible in the above image.

### Increase and decrease in contouring
The effect of contouring increase as we reduce the number of gray levels and the effect decrease as we increase the number of gray levels. They are both vice versa

16 
![Figure 4](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/21-ISO-perference-curves/16.jpg?raw=true)

128
![Figure 5](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/21-ISO-perference-curves/128.jpg?raw=true)

That means more quantization, will effect in more contouring and vice versa. But is this always the case. The answer is No. That depends on something else that is discussed below.

## Isopreference curves
A study conducted on this effect of gray level and contouring, and the results were shown in the graph in the form of curves, known as Iso preference curves.

The phenomena of Isopreference curves shows, that the effect of contouring not only depends on the decreasing of gray level resolution but also on the image detail.

The essence of the study is:

If an image has more detail, the effect of contouring would start appear on this image later, as compare to an image which has less detail, when the gray levels are quantized.

According to the original research, the researchers took these three images and they vary the Gray level resolution, in all three images.

The images were

![Figure 6](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/21-ISO-perference-curves/lena.jpg?raw=true)
![Figure 7](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/21-ISO-perference-curves/cameraman.jpg?raw=true)
![Figure 8](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/21-ISO-perference-curves/public.jpg?raw=true)

### Level of detail
The first image has only a face in it, and hence very less detail. The second image has some other objects in the image too, such as camera man, his camera, camera stand, and background objects e.t.c. Whereas the third image has more details then all the other images.

### Experiment
The gray level resolution was varied in all the images, and the audience was asked to rate these three images subjectively. After the rating, a graph was drawn according to the results.

### Result
The result was drawn on the graph. Each curve on the graph represents one image. The values on the x axis represents the number of gray levels and the values on the y axis represents bits per pixel (k).

The graph has been shown below.

![Figure 9](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/21-ISO-perference-curves/graph.jpg?raw=true)

According to this graph, we can see that the first image which was of face, was subject to contouring early then all of the other two images. The second image, that was of the cameraman was subject to contouring a bit after the first image when its gray levels are reduced. This is because it has more details then the first image. And the third image was subject to contouring a lot after the first two images i-e: after 4 bpp. This is because, this image has more details.

### Conclusion
So for more detailed images, the isopreference curves become more and more vertical. It also means that for an image with a large amount of details, very few gray levels are needed.
