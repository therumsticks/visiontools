import numpy as np
from metrics import IOU

class AnchorFinder:

    def __init__(self, packets):

        self.packets = packets

    def find(self, num_iterations = 20, num_clusters = 5, verbose = False):

        self.num_clusters = num_clusters

        self._set_clusters()

        bounding_boxes = self._get_dimensions()

        for iteration in range(num_iterations):

            ious = []

            # expectation
            for bounding_box in bounding_boxes:

                cluster_index, iou = self._assign_cluster(bounding_box)

                self.clusters[cluster_index].append(bounding_box)

                ious.append(iou)

            # maximization
            for key, item in self.clusters.items():

                    # if a cluster is unassigned, assign a random centroid
                    if len(self.clusters[key]) == 0:

                        self.cluster_centers[key,:] = np.random.randint(0, 300, size = (1, 2)).astype(np.float32)

                    # else assign the new centroid
                    else:

                        self.cluster_centers[key,:] = np.mean(np.vstack((item)), axis=0)

                        self.clusters[key] = list()
            if verbose:

                print "Iteration : {}, IOU Distance : {}, Minimum IOU : {}".format(iteration, np.mean(ious), np.min(ious))

        return self.cluster_centers

    def _set_clusters(self):

        self.cluster_centers = np.random.randint(0, 300, size=(self.num_clusters,2)).astype(np.float32)

        self.clusters = dict()

        for i in range(self.num_clusters):

            self.clusters[i] = list()

    def _get_dimensions(self):

        bounding_boxes = []

        for packet in self.packets:

            for object in packet.objects:

                top_left = object.top_left

                bottom_right = object.bottom_right

                width = bottom_right[0] - top_left[0]

                height = bottom_right[1] - top_left[1]

                bounding_boxes.append([[width, height]])

        return bounding_boxes

    def _assign_cluster(self, bounding_box):

        bounding_box = np.repeat(bounding_box, self.num_clusters, axis=0)

        iou = IOU(bounding_box, self.cluster_centers)

        distance = 1. - iou

        return np.argmin(distance), np.max(iou)

    def get_aspect_ratio(self):

        # caclculate width / height
        return self.cluster_centers[:,0] / self.cluster_centers[:,1]

    def get_area(self):

        # calculate width * height
        return self.cluster_centers[:,0] * self.cluster_centers[:,1]
