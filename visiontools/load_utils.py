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

if __name__ == '__main__':

    packets = load_pascal('/home/therumsticks/Storage/Datasets/VOCdevkit/VOC2007/', mode='trainval')

    packets[0].visualize()
