import matplotlib.image as mplimage
import matplotlib.pyplot as plt

def show_images():
    img_p = mplimage.imread('../data/my_image_0.jpg')
    img_q = mplimage.imread('../data/my_image_1.jpg')

    plt.figure("x0_points")
    plt.imshow(img_p)

    plt.figure("x1_points")
    plt.imshow(img_q)
    plt.show()

show_images()
