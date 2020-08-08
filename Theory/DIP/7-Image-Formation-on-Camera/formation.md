# Image Formation on Camera

## How human eye works?

Before we discuss , the image formation on analog and digital cameras , we have to first discuss the image formation on human eye. Because the basic principle that is followed by the cameras has been taken from the way , the human eye works.

When light falls upon the particular object , it is reflected back after striking through the object. The rays of light when passed through the lens of eye , form a particular angle , and the image is formed on the retina which is the back side of the wall. The image that is formed is inverted. This image is then interpreted by the brain and that makes us able to understand things. Due to angle formation , we are able to perceive the height and depth of the object we are seeing. This has been more explained in the tutorial of perspective transformation.

![Figure 1](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/7-Image-Formation-on-Camera/image_formation_on_camera.jpg?raw=true)

As you can see in the above figure, that when sun light falls on the object (in this case the object is a face), it is reflected back and different rays form different angle when they are passed through the lens and an invert image of the object has been formed on the back wall. The last portion of the figure denotes that the object has been interpreted by the brain and re-inverted.

Now lets take our discussion back to the image formation on analog and digital cameras.

## Image formation on analog cameras

![Figure 2](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/7-Image-Formation-on-Camera/image_formation_using_strip.jpg?raw=true)

In analog cameras , the image formation is due to the chemical reaction that takes place on the strip that is used for image formation.

A 35mm strip is used in analog camera. It is denoted in the figure by 35mm film cartridge. This strip is coated with silver halide ( a chemical substance).

In analog cameras , the image formation is due to the chemical reaction that takes place on the strip that is used for image formation.

![Figure 3](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/7-Image-Formation-on-Camera/analog_strip.jpg?raw=true)

A 35mm strip is used in analog camera. It is denoted in the figure by 35mm film cartridge. This strip is coated with silver halide ( a chemical substance).

Light is nothing but just the small particles known as photon particles.So when these photon particles are passed through the camera, it reacts with the silver halide particles on the strip and it results in the silver which is the negative of the image.

In order to understand it better , have a look at this equation.

Photons (light particles) + silver halide ? silver ? image negative.

![Figure 4](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/7-Image-Formation-on-Camera/silver_halide_negatives.jpg?raw=true)

This is just the basics, although image formation involves many other concepts regarding the passing of light inside , and the concepts of shutter and shutter speed and aperture and its opening but for now we will move on to the next part. Although most of these concepts have been discussed in our tutorial of shutter and aperture.

This is just the basics, although image formation involves many other concepts regarding the passing of light inside , and the concepts of shutter and shutter speed and aperture and its opening but for now we will move on to the next part. Although most of these concepts have been discussed in our tutorial of shutter and aperture.

## Image formation on digital cameras

In the digital cameras , the image formation is not due to the chemical reaction that take place , rather it is a bit more complex then this. In the digital camera , a CCD array of sensors is used for the image formation

1. Image formation through CCD array

![Figure 5](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/7-Image-Formation-on-Camera/ccd_array.jpg?raw=true)

CCD stands for charge-coupled device. It is an image sensor, and like other sensors it senses the values and converts them into an electric signal. In case of CCD it senses the image and convert it into electric signal e.t.c.

This CCD is actually in the shape of array or a rectangular grid. It is like a matrix with each cell in the matrix contains a censor that senses the intensity of photon.

![Figure 6](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/7-Image-Formation-on-Camera/ccd_sensor_array.jpg?raw=true)

Like analog cameras , in the case of digital too , when light falls on the object , the light reflects back after striking the object and allowed to enter inside the camera.

Each sensor of the CCD array itself is an analog sensor. When photons of light strike on the chip , it is held as a small electrical charge in each photo sensor. The response of each sensor is directly equal to the amount of light or (photon) energy striked on the surface of the sensor.

Since we have already define an image as a two dimensional signal and due to the two dimensional formation of the CCD array , a complete image can be achieved from this CCD array.

It has limited number of sensors , and it means a limited detail can be captured by it. Also each sensor can have only one value against the each photon particle that strike on it.

So the number of photons striking(current) are counted and stored. In order to measure accurately these , external CMOS sensors are also attached with CCD array.

## Introduction to pixel
The value of each sensor of the CCD array refers to each the value of the individual pixel. The number of sensors = number of pixels. It also means that each sensor could have only one and only one value.

## Storing image
The charges stored by the CCD array are converted to voltage one pixel at a time. With the help of additional circuits , this voltage is converted into a digital information and then it is stored.

Each company that manufactures digital camera, make their own CCD sensors. That include , Sony , Mistubishi , Nikon ,Samsung , Toshiba , FujiFilm , Canon e.t.c.

Apart from the other factors , the quality of the image captured also depends on the type and quality of the CCD array that has been used.
