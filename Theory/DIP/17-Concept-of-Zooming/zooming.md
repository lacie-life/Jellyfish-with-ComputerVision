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
---------------------------------------------------------------------------------------------------------------

# Zooming methods

## Method 1: Pixel replication

### Introduction

It is also known as Nearest neighbor interpolation. As its name suggest, in this method, we just replicate the neighboring pixels. As we have already discussed in the tutorial of Sampling, that zooming is nothing but increase amount of sample or pixels. This algorithm works on the same principle.

#### Working

In this method we create new pixels form the already given pixels. Each pixel is replicated in this method n times row wise and column wise and you got a zoomed image. Its as simple as that.

### Example

![Figure 3](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/17-Concept-of-Zooming/1.PNG?raw=true)

![Figure 4](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/17-Concept-of-Zooming/2.PNG?raw=true)

### Advantage and disadvantage

One of the advantage of this zooming technique is, it is very simple. You just have to copy the pixels and nothing else.

The disadvantage of this technique is that image got zoomed but the output is very blurry. And as the zooming factor increased, the image got more and more blurred. That would eventually result in fully blurred image.

## Method 2: Zero order hold

### Introduction
Zero order hold method is another method of zooming. It is also known as zoom twice. Because it can only zoom twice. We will see in the below example that why it does that.

### Working
In zero order hold method, we pick two adjacent elements from the rows respectively and then we add them and divide the result by two, and place their result in between those two elements. We first do this row wise and then we do this column wise.

### Example

![Figure 5](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/17-Concept-of-Zooming/3.PNG?raw=true)

### Advantages and disadvantage.
One of the advantage of this zooming technique , that it does not create as blurry picture as compare to the nearest neighbor interpolation method. But it also has a disadvantage that it can only run on the power of 2. It can be demonstrated here.

### Reason behind twice zooming:
Consider the above image of 2 rows and 2 columns. If we have to zoom it 6 times, using zero order hold method , we can not do it. As the formula shows us this.

It could only zoom in the power of 2 2,4,8,16,32 and so on.

Even if you try to zoom it, you can not. Because at first when you will zoom it two times, and the result would be same as shown in the column wise zooming with dimensions equal to 3x3. Then you will zoom it again and you will get dimensions equal to 5 x 5. Now if you will do it again, you will get dimensions equal to 9 x 9.

Whereas according to the formula of yours the answer should be 11x11. As (6(2) minus 1) X (6(2) minus 1) gives 11 x 11.

## Method 3: K-Times zooming

### Introduction:
K times is the third zooming method we are going to discuss. It is one of the most perfect zooming algorithm discussed so far. It caters the challenges of both twice zooming and pixel replication. K in this zooming algorithm stands for zooming factor.

### Working:
It works like this way.

First of all, you have to take two adjacent pixels as you did in the zooming twice. Then you have to subtract the smaller from the greater one. We call this output (OP).

Divide the output(OP) with the zooming factor(K). Now you have to add the result to the smaller value and put the result in between those two values.

Add the value OP again to the value you just put and place it again next to the previous putted value. You have to do it till you place k-1 values in it.

Repeat the same step for all the rows and the columns , and you get a zoomed images.

### Example

![Figure 6](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/17-Concept-of-Zooming/4.PNG?raw=true)

![Figure 7](https://github.com/lacie-life/Image-Processing/blob/master/Theory/DIP/17-Concept-of-Zooming/5.PNG?raw=true)

Advantages and disadvantages
The one of the clear advantage that k time zooming algorithm has that it is able to compute zoom of any factor which was the power of pixel replication algorithm , also it gives improved result (less blurry) which was the power of zero order hold method. So hence It comprises the power of the two algorithms.

The only difficulty this algorithm has that it has to be sort in the end, which is an additional step, and thus increases the cost of computation.


















