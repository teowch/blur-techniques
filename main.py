import sys
import timeit
import numpy as np
import cv2

INPUT_IMAGE = "tree.jpg"

# Here you can change the blur weight.
# This value must be an odd number.
# Note that the larger is the BLUR_SIZE, the longer the algorithm takes.
# Increasing the BLUR_SIZE will show how drastic is the difference in time complexity between the algorithms
BLUR_SIZE = 5

# Here you can choose which algorithms you want to run.
# All you need to do is set to True or False
EXEC = {
    "boxBlur": True,
    "separableBoxBlur": True,
    "integralBlur": True
}


def boxBlur(img, fullw):
    """
    Apply the blur effect using the Convolution Box Blur technique.
    Params:
        img: image that will be blurred
        fullW: blur box size
    Returns:
        blurred image
    """

    halfW = fullw // 2
    img_return = np.zeros_like(img)

    # For each pixel of the image
    for y in range(halfW, len(img) - halfW):
        for x in range(halfW, len(img[0]) - halfW):
            sum = 0

            # Sum each pixel inside the box and get the mean
            for i in range(-halfW, halfW + 1):
                for j in range(-halfW, halfW + 1):
                    sum += img[y + i, x + j]
            mean = sum / (fullw**2)
            img_return[y, x] = mean

    return img_return


def separableBoxBlur(img, fullW):
    """
    Apply the blur effect using the Separable Convolution Box Blur technique.
    Steps:
        1. Blur the image horizontally using a kernel with [1, fullW] dimensions
        2. Blur the image vertically using a kernel with [fullW, 1] dimensions
        3. Blur applied
    Params:
        img: image that will be blurred
        fullW: blur box size
    Returns:
        blurred image
    """

    halfW = fullW // 2

    # Image blurred horizontally
    img_horizontal = np.zeros_like(img)

    # For each pixel of the image
    for y in range(halfW, len(img) - halfW):
        for x in range(halfW, len(img[0]) - halfW):
            sum = 0

            # For each pixel inside the horizontal kernel
            for i in range(-halfW, halfW + 1):
                sum += img[y, x + i]
            media = sum / fullW
            img_horizontal[y, x] = media

    img_return = np.copy(img_horizontal)

    # For each pixel of the image
    for y in range(halfW, len(img) - halfW):
        for x in range(halfW, len(img[0]) - halfW):
            sum = 0

            # For each pixel inside the vertical kernel
            for i in range(-halfW, halfW + 1):
                sum += img_horizontal[y + i, x]
            img_return[y, x] = sum / fullW

    return img_return


def integralBlur(img, fullW):
    """
    Apply the blur effect using an integral image.
    Steps:
        1. Generate the integral image
        2. Get the means
        3. Blur applied
    Params:
        img: image that will be blurred
        fullW: blur box size
    Returns:
        blurred image
    """

    halfW = fullW // 2
    img_return = np.copy(img)
    img_integral = np.zeros_like(img)
    img_integral = img_integral.astype(np.float32)

    # First line of integral image
    for x in range(1, len(img[0])):
        img_integral[0, x] = img[0, x] + img_integral[0, x - 1]

    # First column of integral image
    for y in range(1, len(img)):
        img_integral[y, 0] = img[y, 0] + img_integral[y - 1, 0]

    # Remaining pixels
    for y in range(1, len(img)):
        for x in range(1, len(img[0])):
            img_integral[y, x] = (
                img[y, x]
                + img_integral[y, x - 1]
                + img_integral[y - 1, x]
                - img_integral[y - 1, x - 1]
            )

    # For each pixel of the image
    for y in range(0, len(img)):
        for x in range(0, len(img[0])):
            # Coordinates of the integral image that will be use to calculate the sum and the mean
            xIntegral = x + halfW
            yIntegral = y + halfW
            sum = 0
            boxW = fullW
            boxH = fullW

            # If the pixel is at the right bound
            if x > (len(img[0]) - 1 - halfW):
                xIntegral = len(img[0]) - 1
                boxW = halfW + len(img[0]) - 1 - x

            # If the pixel is at the bottom bound
            if y > (len(img) - 1 - halfW):
                yIntegral = len(img) - 1
                boxH = halfW + len(img) - 1 - y

            sum += img_integral[yIntegral, xIntegral]

            # If the pixel isn't at the top bound
            if y > halfW:
                # Subtract the top right pixel outside the box
                sum -= img_integral[y - halfW - 1, xIntegral]
            else:
                # Adjust the box height
                boxH = y + halfW

            # If the pixel isn't at the left bound
            if x > halfW:
                # Subtract the bottom left pixel outside the box
                sum -= img_integral[yIntegral, x - halfW - 1]
            else:
                # Adjust the box width
                boxW = x + halfW

            # If the pixel is neither at the left bound nor the top bound
            if x > halfW and y > halfW:
                # Sum the top left pixel outside the box
                sum += img_integral[y - halfW - 1, x - halfW - 1]

            mean = sum / (boxH * boxW)
            img_return[y, x] = mean

    return img_return


def main():
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR)

    if img is None:
        print("Cannot open image")
        sys.exit()

    img = img.reshape((img.shape[0], img.shape[1], 3))
    img = img.astype(np.float32)
    img /= 255

    # Algorithms
    if EXEC["boxBlur"]:
        start_time = timeit.default_timer()
        img_output = boxBlur(img, BLUR_SIZE)
        cv2.imwrite("01 - boxBlur.png", img_output * 255)
        print("Box Blur: %.2fs" % (timeit.default_timer() - start_time))
        cv2.imshow("01 - boxBlur", img_output)

    if EXEC["separableBoxBlur"]:
        start_time = timeit.default_timer()
        img_output = separableBoxBlur(img, BLUR_SIZE)
        cv2.imwrite("02 - separableBoxBlur.png", img_output * 255)
        print("Separable Box Blur: %.2fs" % (timeit.default_timer() - start_time))
        cv2.imshow("02 - separableBoxBlur", img_output)

    if EXEC["integralBlur"]:
        start_time = timeit.default_timer()
        img_output = integralBlur(img, BLUR_SIZE)
        cv2.imwrite("03 - integralBlur.png", img_output * 255)
        print("Integral Blur: %.2fs" % (timeit.default_timer() - start_time))
        cv2.imshow("03 - integralBlur", img_output)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
