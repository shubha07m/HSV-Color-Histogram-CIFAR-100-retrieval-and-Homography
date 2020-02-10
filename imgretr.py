import cv2 as cv
from matplotlib import pyplot as plt
import glob


def myhistogram(x):
    x = cv.imread(x)
    height = x.shape[0]
    width = x.shape[1]
    pixels = height * width
    b, g, r = cv.split(x)

    hist1 = cv.calcHist([b, g, r], [0], None, [pixels], [0, pixels])
    hist2 = cv.calcHist([b, g, r], [1], None, [pixels], [0, pixels])
    hist3 = cv.calcHist([b, g, r], [2], None, [pixels], [0, pixels])

    cv.imshow('b', b)
    cv.imshow('g', g)
    cv.imshow('r', r)

    # b, g, r =
    # return (b == hist1), (g == hist2), (r == hist3)


print(myhistogram('1.png'))

cv.waitKey(0)
cv.destroyAllWindows()
