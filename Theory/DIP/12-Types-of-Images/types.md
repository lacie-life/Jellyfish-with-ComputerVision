# Types of Images

There are many type of images, and we will look in detail about different types of images, and the color distribution in them.

## The binary image

The binary image as it name states, contain only two pixel values.

0 and 1.

In our previous tutorial of bits per pixel, we have explained this in detail about the representation of pixel values to their respective colors.

Here 0 refers to black color and 1 refers to white color. It is also known as Monochrome.

## Black and white image

The resulting image that is formed hence consist of only black and white color and thus can also be called as Black and White image.

![Figure 1](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/12-Types-of-Images/black_and_white.jpg?raw=true)

### No gray level

One of the interesting this about this binary image that there is no gray level in it. Only two colors that are black and white are found in it

### Format

Binary images have a format of PBM ( Portable bit map )

## 2, 3, 4,5, 6 bit color format

The images with a color format of 2, 3, 4, 5 and 6 bit are not widely used today. They were used in old times for old TV displays, or monitor displays.

But each of these colors have more then two gray levels, and hence has gray color unlike the binary image.

In a 2 bit 4, in a 3 bit 8, in a 4 bit 16, in a 5 bit 32, in a 6 bit 64 different colors are present.

## 8 bit color format

8 bit color format is one of the most famous image format. It has 256 different shades of colors in it. It is commonly known as Grayscale image.

The range of the colors in 8 bit vary from 0-255. Where 0 stands for black, and 255 stands for white, and 127 stands for gray color.

This format was used initially by early models of the operating systems UNIX and the early color Macintoshes.

A grayscale image of Einstein is shown below:

![Figure 2](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/12-Types-of-Images/einstein.jpg?raw=true)

### Format 
The format of these images are PGM ( Portable Gray Map ).

This format is not supported by default from windows. In order to see gray scale image, you need to have an image viewer or image processing toolbox such as Matlab.

### Behind gray scale image
As we have explained it several times in the previous tutorials, that an image is nothing but a two dimensional function, and can be represented by a two dimensional array or matrix. So in the case of the image of Einstein shown above, there would be two dimensional matrix in behind with values ranging between 0 and 255.

But thats not the case with the color images.

## 16 bit color format

It is a color image format. It has 65,536 different colors in it. It is also known as High color format.

It has been used by Microsoft in their systems that support more then 8 bit color format. Now in this 16 bit format and the next format we are going to discuss which is a 24 bit format are both color format.

The distribution of color in a color image is not as simple as it was in grayscale image.

A 16 bit format is actually divided into three further formats which are Red , Green and Blue. The famous (RGB) format.

It is pictorially represented in the image below.

![Figure 3](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/12-Types-of-Images/16-bit.jpg?raw=true)

Now the question arises, that how would you distribute 16 into three. If you do it like this,

5 bits for R, 5 bits for G, 5 bits for B

Then there is one bit remains in the end.

So the distribution of 16 bit has been done like this.

5 bits for R, 6 bits for G, 5 bits for B.

The additional bit that was left behind is added into the green bit. Because green is the color which is most soothing to eyes in all of these three colors.

Note this is distribution is not followed by all the systems. Some have introduced an alpha channel in the 16 bit.

### Another distribution of 16 bit format is like this:
4 bits for R, 4 bits for G, 4 bits for B, 4 bits for alpha channel.

Or some distribute it like this

5 bits for R, 5 bits for G, 5 bits for B, 1 bits for alpha channel.

## 24 bit color format

24 bit color format also known as true color format. Like 16 bit color format, in a 24 bit color format, the 24 bits are again distributed in three different formats of Red, Green and Blue.

![Figure 4](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/12-Types-of-Images/24-bit.jpg?raw=true)

Since 24 is equally divided on 8, so it has been distributed equally between three different color channels.

Their distribution is like this.

8 bits for R, 8 bits for G, 8 bits for B.

### Behind a 24 bit image.

Unlike a 8 bit gray scale image, which has one matrix behind it, a 24 bit image has three different matrices of R, G, B.

![Figure 4](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/12-Types-of-Images/what_is_image.jpg?raw=true)

### Format
It is the most common used format. Its format is PPM ( Portable pixMap) which is supported by Linux operating system. The famous windows has its own format for it which is BMP ( Bitmap ).
