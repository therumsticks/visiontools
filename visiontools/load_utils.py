import xml.etree.ElementTree as ET

from objects import Object
from packet import Packet

def load_pascal(path, mode='train'):

    packets = []

    imageset_path = path + "ImageSets/Main/" + mode + ".txt"

    annotation_base_path = path + "Annotations/"

    image_base_path = path + "JPEGImages/"

    with open(imageset_path, 'rb') as f:

        file_list = [x.strip() for x in f.readlines()]

        for file_name in file_list:

            file_path = annotation_base_path + file_name + '.xml'

            image_path = image_base_path + file_name + '.jpg'

            tree = ET.parse(file_path)

            root = tree.getroot()

            object_list = []

            for object in root.iter('object'):

                category = object.find('name').text

                xmin = float(object.find('bndbox').find('xmin').text)

                ymin = float(object.find('bndbox').find('ymin').text)

                xmax = float(object.find('bndbox').find('xmax').text)

                ymax = float(object.find('bndbox').find('ymax').text)

                obj = Object(category, top_left = (xmin, ymin), bottom_right = (xmax, ymax))

                object_list.append(obj)

            packet = Packet(image_path = image_path, objects = object_list)

            packets.append(packet)

    return packets

def load_kitti(path, mode='train', image_dir=None, label_dir=None):

    packets = []

    with open(path + mode + '.txt', 'rb') as f:

        file_list = [x.strip() for x in f.readlines()]

    for file in file_list:

        image_path = image_dir + file + '.png'

        label_path = label_dir + file + '.txt'

        with open(label_path) as f:

            object_list = []

            for line in f:

                obj_class, truncated, occluded, alpha, bx1, by1, bx2, by2, dz, dy, dx, tx, ty, tz, rot_y = line.split()

                if obj_class == 'Car': #and float(truncated) == 0: #and float(occluded) == 0 :

                    size = [float(x) for x in [dx, dy, dz]]

                    position = [float(x) for x in [tx, ty, tz]]

                    rot_y = float(rot_y)

                    top_left = (int(float(bx1)), int(float(by1)))

                    bottom_right = (int(float(bx2)), int(float(by2)))

                    obj = Object(obj_class, top_left = top_left, bottom_right = bottom_right)

                    obj.size = size

                    obj.position = position

                    object_list.append(obj)

            if object_list != []:

                packet = Packet(image_path, objects = object_list)

            else:

                continue

        packets.append(packet)

    return packets

if __name__ == '__main__':

    packets = load_pascal('/home/therumsticks/Storage/Datasets/VOCdevkit/VOC2007/', mode='trainval')

    packets[0].visualize()


    dataset_dir = '/media/vision/New Volume/Multibin/'
    training_dir = dataset_dir + 'training_data/'
    image_dir = training_dir + 'images/'
    label_dir = training_dir + 'labels/'

    packets = load_kitti("/media/vision/New Volume/Projects/Deep3D/data/", mode="train", image_dir=image_dir, label_dir=label_dir)

    packets[0].visualize()
