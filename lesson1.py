import cv2
import numpy as np
import math


def ex1():
    # hello world: open image and display it in a window
    image = cv2.imread('./4.2.04.tiff')
    cv2.imshow('Hello World', image)
    k = cv2.waitKey(0)

def ex2_1():
    image = cv2.imread('./4.2.04.tiff')
    image[:, :, 0] = 0
    cv2.imshow('Hello World', image)
    k = cv2.waitKey(0)

def ex2_2():
    image = cv2.imread('./4.2.04.tiff')
    image[:, :, 2] = 0
    cv2.imshow('Hello World', image)
    k = cv2.waitKey(0)

def ex2_3():
    image = cv2.imread('./4.2.04.tiff')
    new_image = np.zeros_like(image)
    new_image[0:256, 0:256] = image[256:512, 0:256]
    new_image[256:512, 0:256] = image[256:512, 256:512]
    new_image[256:512, 256:512] = image[0:256, 256:512]
    new_image[0:256, 256:512] = image[0:256, 0:256]
    cv2.imshow('Hello World', new_image)
    k = cv2.waitKey(0)

def ex3():
    image = cv2.imread('./4.2.04.tiff')
    new_image = np.zeros_like(image)
    new_image[0:256, 0:256] = image[256:512, 0:256]
    new_image[256:512, 0:256] = image[256:512, 256:512]
    new_image[256:512, 256:512] = image[0:256, 256:512]
    new_image[0:256, 256:512] = image[0:256, 0:256]
    # OpenCV automatically converts to the correct output format
    cv2.imwrite('out.png', new_image)

def ex4_1():
    image = np.zeros([400, 400, 3])
    image = cv2.circle(image, (0, 0), 50, (255, 0, 0), 1)
    image = cv2.circle(image, (200, 200), 50, (0, 255, 0), 1)
    image = cv2.circle(image, (400, 400), 50, (0, 0, 255), 1)
    cv2.imshow('Hello World', image)
    k = cv2.waitKey(0)

def ex4_2():
    image = np.zeros([400, 400, 3])

    for i in range(20):
        pos = tuple(np.random.random_integers(400, size=2))
        radius = np.random.random_integers(5, 50)
        color = np.random.uniform(size=3)
        image = cv2.circle(image, pos, radius, color, 1)

    cv2.imshow('Hello World', image)
    k = cv2.waitKey(0)

def ex4_3():
    image = np.zeros([400, 400, 3])

    for i in range(20):
        pos1 = tuple(np.random.random_integers(400, size=2))
        pos2 = tuple(np.random.random_integers(400, size=2))
        color = np.random.uniform(size=3)
        image = cv2.line(image, pos1, pos2, color, 1)

        cv2.imshow('Hello World', image)
        k = cv2.waitKey(0)

def ex5():
    KEY_ARROW_UP = 63232
    KEY_ARROW_LEFT = 63234
    KEY_ARROW_DOWN = 63233
    KEY_ARROW_RIGHT = 63235
    KEY_QUIT = 27
    
    pos = [200, 200]
    while True:
        image = np.zeros([400, 400, 3])
        image = cv2.circle(image, tuple(pos), 10, (0, 0, 255), 1)
        cv2.imshow('Hello World', image)
        k = cv2.waitKey(0)

        if k == KEY_QUIT:
            break
        elif k == KEY_ARROW_LEFT:
            pos[0] -= 10
        elif k == KEY_ARROW_RIGHT:
            pos[0] += 10
        elif k == KEY_ARROW_UP:
            pos[1] -= 10
        elif k == KEY_ARROW_DOWN:
            pos[1] += 10

def ex6():
    KEY_QUIT = 27
    vc = cv2.VideoCapture(0)
    
    while True:
        ret, image = vc.read()
        cv2.imshow('Frame', image)
        k = cv2.waitKey(16)
        if k == KEY_QUIT:
            break
    vc.release()
    cv2.destroyAllWindows()

def ex7():
    KEY_QUIT = 27
    vc = cv2.VideoCapture(0)

    image = cv2.imread('./4.2.04.tiff')
    cv2.imshow('Static', image)
    cv2.moveWindow('Static', 0, 0)

    while True:
        ret, frame = vc.read()
        cv2.imshow('Frame', frame)
        cv2.moveWindow('Frame', 400, 0)
        k = cv2.waitKey(16)
        if k == KEY_QUIT:
            break
    vc.release()
    cv2.destroyAllWindows()

def magnitude(image, pos):
    x, y = pos
    return np.mean(image[x, y])

def isPeak(image, pos, threshold):
    x, y = pos
    mag = magnitude(image, pos)
    for i in range(x-1, x+1):
        for j in range(y-1, y+1):
            if i == x and j == y:
                continue
            elif (magnitude(image, (i, j)) / mag) > (1 - threshold) :
                return False
    return True

def feature_detector(image, threshold):
    features = []
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            if isPeak(image, (i, j), threshold):
                features.append((i, j))
    return features

# image = cv2.imread('./small.tiff')
# features = feature_detector(image, 0.1)

# for (y, x) in features:
#     image = cv2.circle(image, (x, y), 2, (0, 0, 0), 0)

# cv2.imshow('features', image)
# k = cv2.waitKey(0)

image = cv2.imread('4.2.03.tiff')
cv2.imshow('Window Name', image)
cv2.waitKey(0)
