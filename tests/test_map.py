from context import visiontools
from visiontools.load_utils import load_kitti
from visiontools.metrics import MeanAveragePrecision
from visiontools.objects import Object

import copy

dataset_dir = '/media/vision/New Volume/Multibin/'
training_dir = dataset_dir + 'training_data/'
image_dir = training_dir + 'images/'
label_dir = training_dir + 'labels/'

packets = load_kitti("/media/vision/New Volume/Projects/Deep3D/data/", mode="train", image_dir=image_dir, label_dir=label_dir)

# no false positive, 1 false negative
prediction = copy.deepcopy(packets[1])
target = copy.deepcopy(packets[1])

for i in range(3):

    prediction.objects[i].confidence = 1.

prediction.objects.pop()

mAP = MeanAveragePrecision(prediction, target)

print mAP.calculate()

# one false negative and one false positive
prediction = copy.deepcopy(packets[1])
target = copy.deepcopy(packets[1])

for i in range(3):

    prediction.objects[i].confidence = 1.

prediction.objects.pop()
obj = Object('car', (0.,0), (1,1.))
obj.confidence = 1.
prediction.objects.append(obj)

mAP = MeanAveragePrecision(prediction, target)

print mAP.calculate()
