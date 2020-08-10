# Concept of Zooming

In this tutorial we are going to introduce the concept of zooming, and the common techniques that are used to zoom an image.

## Zooming

Zooming simply means enlarging a picture in a sense that the details in the image became more visible and clear. Zooming an image has many wide applications ranging from zooming through a camera lens, to zoom an image on internet e.t.c.

For example

![Figure 1](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/17-Concept-of-Zooming/einstein.jpg?raw=true)

is zoomed into

![Figure 2](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/17-Concept-of-Zooming/einsteinzoomed.jpg?raw=true)

You can zoom something at two different steps.

The first step includes zooming before taking an particular image. This is known as pre processing zoom. This zoom involves hardware and mechanical movement.

The second step is to zoom once an image has been captured. It is done through many different algorithms in which we manipulate pixels to zoom in the required portion.

We will discuss them in detail in the next tutorial.

## Optical Zoom vs Digital Zoom

### Optical Zoom
The optical zoom is achieved using the movement of the lens of your camera. An optical zoom is actually a true zoom. The result of the optical zoom is far better then that of digital zoom. In optical zoom, an image is magnified by the lens in such a way that the objects in the image appear to be closer to the camera. In optical zoom the lens is physically extend to zoom or magnify an object.

### Digital Zoom
Digital zoom is basically image processing within a camera. During a digital zoom, the center of the image is magnified and the edges of the picture got crop out. Due to magnified center, it looks like that the object is closer to you.

During a digital zoom, the pixels got expand , due to which the quality of the image is compromised.

The same effect of digital zoom can be seen after the image is taken through your computer by using an image processing toolbox / software, such as Photoshop.

The following picture is the result of digital zoom done through one of the following methods given below in the zooming methods.

![Figure 2](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/17-Concept-of-Zooming/digital_zoom.jpg?raw=true)

Now since we are leaning digital image processing, we will not focus, on how an image can be zoomed optically using lens or other stuff. Rather we will focus on the methods, that enable to zoom a digital image.

## Zooming methods:
Although there are many methods that does this job, but we are going to discuss the most common of them here.

They are listed below.

+ Pixel replication or (Nearest neighbor interpolation)
+ Zero order hold method
+ Zooming K times
All these three methods are formally introduced in the next tutorial.