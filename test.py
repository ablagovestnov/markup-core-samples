import numpy as np

from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist

import math


class PointsClusterer():
    points = np.array([[33., 41.],
                       [33.9693, 41.3923],
                       [33.6074, 41.277],
                       [34.4823, 41.919],
                       [34.3702, 41.1424],
                       [34.3931, 41.078],
                       [34.2377, 41.0576],
                       [34.2395, 41.0211],
                       [34.4443, 41.3499],
                       [34.3812, 40.9793]])

    def distance(self, origin, destination):  # found here https://gist.github.com/rochacbruno/2883505
        # lat1, lon1 = origin[0],origin[1]
        # lat2, lon2 = destination[0],destination[1]
        # radius = 6371 # km
        # dlat = math.radians(lat2-lat1)
        # dlon = math.radians(lon2-lon1)
        # a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        #     * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
        # c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        # d = radius * c
        d = abs(origin[1] - destination[1])
        return d

    def create_clusters(self, number_of_clusters, points):
        kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(points)
        l_array = np.array([[label] for label in kmeans.labels_])
        clusters = np.append(points, l_array, axis=1)
        return clusters

    def validate_solution(self, max_dist, clusters):
        _, __, n_clust = clusters.max(axis=0)
        n_clust = int(n_clust)
        for i in range(n_clust):
            two_d_cluster = clusters[clusters[:, 2] == i][:, np.array([True, True, False])]
            if not self.validate_cluster(max_dist, two_d_cluster):
                return False
            else:
                continue
        return True

    def validate_cluster(self, max_dist, cluster):
        distances = cdist(
            cluster,
            cluster,
            lambda ori, des: int(round(self.distance(ori, des)))
        )
        print(distances)
        print(30 * '-')
        for item in distances.flatten():
            if item > max_dist:
                return False
        return True

    def do_cluster(self):
        for i in range(2, len(self.points)):
            print(i)
            print(self.validate_solution(10, self.create_clusters(i, self.points)))

    def my_cluster(self, distance):
        coords = self.arr_points
        C = []
        while len(coords):
            locus = coords.pop()
            cluster = [x for x in coords if abs(locus -x) <= distance]
            C.append(cluster + [locus])
            for x in cluster:
                coords.remove(x)
        return C

    def __init__(self, points=[]):
        self.points = np.array(points)
        self.arr_points = points