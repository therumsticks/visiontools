from objects import Object

import visdom
import cv2
import numpy as np

vis = visdom.Visdom()

class Packet:

    def __init__(self, image_path, objects, size = (None, None)):

        self.image_path = image_path

        self.objects = objects

        self.size = size

    def visualize(self, win = 'image1'):

        image = self.load_image()

        for object in self.objects:

            cv2.rectangle(image, object.top_left, object.bottom_right, color = (0, 255, 0))

        image = np.transpose(image, (2, 0, 1))

        vis.image(image, win = win)

    def load_image(self, mode = 'rgb'):

        image = cv2.imread(self.image_path)

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if mode == 'rgb' else image
