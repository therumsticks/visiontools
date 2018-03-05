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


def IOU_by_dims(box, box_):

    num_boxes = box.shape[0]

    # calculate intersection topleft, bottomright
    topleft_x = np.max(np.vstack((box[:,0], box_[:,0])).T, axis=1).reshape(-1,1)

    topleft_y = np.max(np.vstack((box[:,1], box_[:,1])).T, axis=1).reshape(-1,1)

    bottomright_x = np.min(np.vstack((box[:,2], box_[:,2])).T, axis=1).reshape(-1,1)

    bottomright_y = np.min(np.vstack((box[:,3], box_[:,3])).T, axis=1).reshape(-1,1)

    # calculate intersection area
    width = np.max(np.hstack((np.zeros((num_boxes, 1)), bottomright_x - topleft_x)), axis=1).reshape(-1, 1)
    height = np.max(np.hstack((np.zeros((num_boxes, 1)), bottomright_y - topleft_y)), axis=1).reshape(-1, 1)
    intersection = width * height

    # calcaulate area of the two bounding boxes
    area_box = area(box)
    area_box_ = area(box_)

    # calculate the union area of the two boxes
    union = area_box + area_box_ - intersection

    # intersection over union
    iou = intersection / union

    return iou

def area(box):

    # area of bounding box
    return ((box[:,2] - box[:,0]) * (box[:,3] - box[:,1])).reshape(-1, 1)


if __name__ == '__main__':

    # tests
    box1 = np.array([[0., 0., 1, 1], [100, 100, 200, 200], [200, 200, 300, 400]])
    box2 = np.array([[0.5, 0.5, 1.5, 1.5], [1, 1, 2, 2], [225, 225, 300, 400]])

    print IOU_by_dims(box1,box2)
