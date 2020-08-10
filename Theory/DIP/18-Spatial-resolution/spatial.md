# Spatial resolution

## Image resolution
Image resolution can be defined in many ways. One type of it which is pixel resolution that has been discussed in the tutorial of pixel resolution and aspect ratio.

In this tutorial, we are going to define another type of resolution which is spatial resolution.

## Spatial resolution
Spatial resolution states that the clarity of an image cannot be determined by the pixel resolution. The number of pixels in an image does not matter.

Spatial resolution can be defined as the

smallest discernible detail in an image. (Digital Image Processing - Gonzalez, Woods - 2nd Edition)

Or in other way we can define spatial resolution as the number of independent pixels values per inch.

In short what spatial resolution refers to is that we cannot compare two different types of images to see that which one is clear or which one is not. If we have to compare the two images, to see which one is more clear or which has more spatial resolution, we have to compare two images of the same size.

For example:

You cannot compare these two images to see the clarity of the image.

![Figure 1](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/18-Spatial-resolution/einstein(2).jpg?raw=true)

![Figure 2](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/18-Spatial-resolution/einsteinzoomed.jpg?raw=true)

Although both images are of the same person, but that is not the condition we are judging on. The picture on the left is zoomed out picture of Einstein with dimensions of 227 x 222. Whereas the picture on the right side has the dimensions of 980 X 749 and also it is a zoomed image. We cannot compare them to see that which one is more clear. Remember the factor of zoom does not matter in this condition, the only thing that matters is that these two pictures are not equal.

So in order to measure spatial resolution , the pictures below would server the purpose.

![Figure 3](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/18-Spatial-resolution/einstein.jpg?raw=true)

![Figure 4](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/18-Spatial-resolution/einstein_spatial.jpg?raw=true)

Now you can compare these two pictures. Both the pictures has same dimensions which are of 227 X 222. Now when you compare them, you will see that the picture on the left side has more spatial resolution or it is more clear then the picture on the right side. That is because the picture on the right is a blurred image.

### Measuring spatial resolution
Since the spatial resolution refers to clarity, so for different devices, different measure has been made to measure it.

### For example
+ Dots per inch
+ Lines per inch
+ Pixels per inch
They are discussed in more detail in the next tutorial but just a brief introduction has been given below.

### Dots per inch
Dots per inch or DPI is usually used in monitors.

### Lines per inch
Lines per inch or LPI is usually used in laser printers.

### Pixel per inch
Pixel per inch or PPI is measure for different devices such as tablets , Mobile phones e.t.c.

--------------------------------------------------------------------------------------

## Pixels per inch

Pixel density or Pixels per inch is a measure of spatial resolution for different devices that includes tablets, mobile phones.

The higher is the PPI, the higher is the quality. In order to more understand it, that how it calculated. Lets calculate the PPI of a mobile phone.

![Figure 5](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/18-Spatial-resolution/1.PNG?raw=true)

## Dots per inch

The dpi is often relate to PPI, whereas there is a difference between these two. DPI or dots per inch is a measure of spatial resolution of printers. In case of printers, dpi means that how many dots of ink are printed per inch when an image get printed out from the printer.

Remember, it is not necessary that each Pixel per inch is printed by one dot per inch. There may be many dots per inch used for printing one pixel. The reason behind this that most of the color printers uses CMYK model. The colors are limited. Printer has to choose from these colors to make the color of the pixel whereas within pc, you have hundreds of thousands of colors.

The higher is the dpi of the printer, the higher is the quality of the printed document or image on paper.

Usually some of the laser printers have dpi of 300 and some have 600 or more.

## Lines per inch

When dpi refers to dots per inch, liner per inch refers to lines of dots per inch. The resolution of halftone screen is measured in lines per inch.

The following table shows some of the lines per inch capacity of the printers.

|Printer|LPI|
|-------|----|
|Screen printing|45-65 lpi|
|Laser printer (300dpi)|65 lpi|
|Laser printer (600dpi)|85-100 lpi|
|Offset Press (newsprint paper)|85 lpi|
|Offset Press (coated paper)|85-185 lpi|

