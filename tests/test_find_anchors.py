from context import vision_tools
from vision_tools.load_utils import load_pascal
from vision_tools.find_anchors import AnchorFinder


packets = load_pascal('/home/therumsticks/Storage/Datasets/VOCdevkit/VOC2007/', mode='trainval')

anchor_finder = AnchorFinder(packets)

anchor_finder.find(verbose=True)
