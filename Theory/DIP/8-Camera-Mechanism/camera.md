# Camera Mechanism

In this tutorial, we will discuss some of the basic camera concepts, like aperture, shutter, shutter speed, ISO and we will discuss the collective use of these concepts to capture a good image.

## Aperture

Aperture is a small opening which allows the light to travel inside into camera. Here is the picture of aperture.

![Figure 1](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/8-Camera-Mechanism/aperture.jpg?raw=true)

You will see some small blades like stuff inside the aperture. These blades create a octagonal shape that can be opened closed. And thus it make sense that, the more blades will open, the hole from which the light would have to pass would be bigger. The bigger the hole, the more light is allowed to enter.

### Effect 

The effect of the aperture directly corresponds to brightness and darkness of an image. If the aperture opening is wide, it would allow more light to pass into the camera. More light would result in more photons, which ultimately result in a brighter image.

The example of this is shown below

Consider these two photos

![Figure 2](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/8-Camera-Mechanism/einstein_bright.jpg?raw=true)

![Figure 3](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/8-Camera-Mechanism/einstein_dark.jpg?raw=true)

The one on the right side looks brighter, it means that when it was captured by the camera, the aperture was wide open. As compare to the other picture on the left side, which is very dark as compare to the first one, that shows that when that image was captured, its aperture was not wide open.

### Size 

Now lets discuss the maths behind the aperture. The size of the aperture is denoted by a f value. And it is inversely proportional to the opening of aperture.

Here are the two equations, that best explain this concept.

Large aperture size = Small f value

Small aperture size = Greater f value

Pictorially it can be represented as:

![Figure 4](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/8-Camera-Mechanism/focal.jpg?raw=true)

## Shutter 

After the aperture, there comes the shutter. The light when allowed to pass from the aperture, falls directly on to the shutter. Shutter is actually a cover, a closed window, or can be thought of as a curtain. Remember when we talk about the CCD array sensor on which the image is formed. Well behind the shutter is the sensor. So shutter is the only thing that is between the image formation and the light, when it is passed from aperture.

As soon as the shutter is open, light falls on the image sensor, and the image is formed on the array.

### Effect

If the shutter allows light to pass a bit longer, the image would be brighter. Similarly a darker picture is produced, when a shutter is allowed to move very quickly and hence, the light that is allowed to pass has very less photons, and the image that is formed on the CCD array sensor is very dark.

Shutter has further two main concepts:

- Shutter Speed
- Shutter time


### Shutter speed

The shutter speed can be referred to as the number of times the shutter get open or close. Remember we are not talking about for how long the shutter get open or close.
(what is different between the number of times get open/close and how long get open/close???????)

### Shutter time

The shutter time can be defined as

When the shutter is open, then the amount of wait time it take till it is closed is called shutter time.

In this case we are not talking about how many times, the shutter got open or close, but we are talking about for how much time does it remain wide open.

For example:

We can better understand these two concepts in this way. That lets say that a shutter opens 15 times and then get closed, and for each time it opens for 1 second and then get closed. In this example, 15 is the shutter speed and 1 second is the shutter time.

### Relationship

The relationship between shutter speed and shutter time is that they are both inversely proportional to each other.

This relationship can be defined in the equation below.

More shutter speed = less shutter time.

Less shutter speed = more shutter time.

Explanation:
The lesser the time required, the more is the speed. And the greater the time required, the less is the speed.

## Applications

These two concepts together make a variety of applications. Some of them are given below

### Fast ,oving objects:

If you were to capture the image of a fast moving object, could be a car or anything. The adjustment of shutter speed and its time would effect a lot.

So, in order to capture an image like this, we will make two amendments:
+ Increase shutter speed
+ Decrease shutter time
What happens is, that when we increase shutter speed, the more number of times, the shutter would open or close. It means different samples of light would allow to pass in. And when we decrease shutter time, it means we will immediately captures the scene, and close the shutter gate.

If you will do this, you get a crisp image of a fast moving object.

In order to understand it, we will look at this example. Suppose you want to capture the image of fast moving water fall.

You set your shutter speed to 1 second and you capture a photo. This is what you get

![Figure 5](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/8-Camera-Mechanism/one_sec.jpg?raw=true)

Then you set your shutter speed to a faster speed and you get.

![Figure 6](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/8-Camera-Mechanism/one_by_three_sec.jpg?raw=true)

Then again you set your shutter speed to even more faster and you get.

![Figure 7](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/8-Camera-Mechanism/one_by_two_hundred.jpg?raw=true)

You can see in the last picture, that we have increase our shutter speed to very fast, that means that a shutter get opened or closed in 200th of 1 second and so we got a crisp image.

## ISO

ISO factor is measured in numbers. It denotes the sensitivity of light to camera. If ISO number is lowered, it means our camera is less sensitive to light and if the ISO number is high, it means it is more sensitive.

### Effect
The higher is the ISO, the more brighter the picture would be. IF ISO is set to 1600, the picture would be very brighter and vice versa.

### Side effect
If the ISO increases, the noise in the image also increases. Today most of the camera manufacturing companies are working on removing the noise from the image when ISO is set to higher speed.