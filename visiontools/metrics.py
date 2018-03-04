import numpy as np

def IOU(box, box_):
    """
    Args:

        box : numpy array (n,2), rows are different boxes, columns are
            (width, height)
        box1 : same as box

    Returns:

        iou : numpy array (n,1) Intersection over union of the corresponding boxes in the two numpy
        arrays
    """

    min_width = np.min(np.vstack((box[:,0], box_[:,0])).T, axis=1).reshape(-1, 1)

    max_width = np.max(np.vstack((box[:,0], box_[:,0])).T, axis=1).reshape(-1, 1)

    min_height = np.min(np.vstack((box[:,1], box_[:,1])).T, axis=1).reshape(-1, 1)

    max_height = np.max(np.vstack((box[:,1], box_[:,1])).T, axis=1).reshape(-1, 1)

    intersection = min_width * min_height

    area_box = (box[:,0] * box[:,1]).reshape(-1, 1)

    area_box_ = (box_[:,0] * box_[:,1]).reshape(-1, 1)

    union = area_box + area_box_ - intersection

    iou = intersection / union
    
    return iou


if __name__ == '__main__':

    box1 = np.array([[10., 15]])
    box2 = np.array([[15, 10.]])

    print IOU(box1,box2)
