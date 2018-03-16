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

class MeanAveragePrecision:
    """
    Calculates mean average precision
    Caveats : NMS already performed, Class Thresholding already performed, only
            single class allowed
    """

    def __init__(self, prediction, target):

        self.prediction = prediction

        self.target = target

    def calculate(self):
        """
        This method returns the precision and recall
        """
        # sort predictions based on confidence
        # haven't used confidence thresholding. Input only thresholded preds
        objects = self._sort(self.prediction.objects)

        # for every object extract the bounding box and save it in set
        prediction_objects = self._to_set(objects)

        target_objects = self._to_set(self.target.objects)

        # calculate precision recall
        precision, recall = self._average_precision(prediction_objects, target_objects)

        # calculate the average precision

        return precision, recall

    def _sort(self, objects):
        """
        This function sorts the bounding box detections in reverse order based
        on the confidence
        """
        objects.sort(key=lambda x: x.confidence, reverse=True)
        return objects

    def _to_set(self, objects):

        object_set = set()

        for object in objects:

            xmin, ymin = object.top_left

            xmax, ymax = object.bottom_right

            object_set.add((xmin, ymin, xmax, ymax))

        return object_set

    def _average_precision(self, predictions, targets):

        epsilon = 1e-6

        true_positive, false_positive, true_negative, false_negative = [0., 0., 0., 0.]

        # find the max overlap (IOU) prediction
        for prediction_box in predictions.copy():

            iou, max_overlap_box = self._find_box(prediction_box, targets)

            # metrics for average precision calculated here
            # match found
            if iou > 0.5:

                true_positive += 1.

                #remove the ground truth box whose match has been found
                targets.remove(max_overlap_box)

            # not a match
            else:

                # ground truth box is not removed as it wasn't a true positive
                false_positive += 1.

            # remove the prediction box irrespective because the best match has
            # already been found. It may or may not be a good prediction
            predictions.remove(prediction_box)


        # remaining items in prediction set are false positives
        false_positive += len(predictions)

        # remaining items in target set are false negatives
        false_negative += len(targets)

        precision = true_positive / (true_positive + false_positive + epsilon)

        recall = true_positive / (true_positive + false_negative)

        return precision, recall

    def _find_box(self, prediction_box, targets):

        max_iou = 0.

        max_overlap_box = None

        for target_box in targets:

            # calculate IOU
            iou = IOU_by_dims(np.array([prediction_box]), np.array([target_box]))

            if iou > max_iou:

                max_iou = iou

                max_overlap_box = target_box

        return max_iou, max_overlap_box


if __name__ == '__main__':

    # tests
    box1 = np.array([[0., 0., 1, 1], [100, 100, 200, 200], [200, 200, 300, 400]])
    box2 = np.array([[0.5, 0.5, 1.5, 1.5], [1, 1, 2, 2], [225, 225, 300, 400]])

    print IOU_by_dims(box1,box2)
