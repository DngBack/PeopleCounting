from enum import IntEnum
from collections import namedtuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.misc import AverageEstimator


THE_BIGGEST_DISTANCE = 2.0

TrackedObj = namedtuple("TrackedObj", ["rect", "label", "display"])


def euclidean_distance(x, y, squared=False):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x.shape) == 1 and len(y.shape) == 1:
        if squared:
            return np.sum((x - y) ** 2)
        else:
            return np.sqrt(np.sum((x - y) ** 2))
    else:
        xx = (x * x).sum(axis=1)[:, np.newaxis]
        yy = (y * y).sum(axis=1)[np.newaxis, :]
        squared_dist = xx + yy - 2 * x @ y.T
        squared_dist = np.maximum(squared_dist, 0)
        if squared:
            return squared_dist
        else:
            return np.sqrt(squared_dist)


def cosine_distance(a, b, data_is_normalized=False):
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a.shape) == 1 and len(b.shape) == 1:
        if not data_is_normalized:
            a = a / np.linalg.norm(a, axis=0)
            b = b / np.linalg.norm(b, axis=0)
        return 1.0 - np.dot(a, b)
    else:
        if not data_is_normalized:
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            b = b / np.linalg.norm(b, axis=1, keepdims=True)
        return 1.0 - np.dot(a, b.T)


# Lớp này được dùng để phân cụm các đặc trưng một cachs tự động và linh hoạt
class ClusterFeature:
    def __init__(self, feature_len, init_dis_thres=0.1):
        self.clusters = []
        self.clusters_sizes = []
        self.feature_len = feature_len
        self.init_dis_thres = init_dis_thres
        self.global_merge_weight = 0.2

    def update(self, feature_vec, num=1):
        if len(self.clusters) == 0:
            self.clusters.append(feature_vec)
            self.clusters_sizes.append(num)
        elif len(self.clusters) < self.feature_len:
            distances = cosine_distance(
                feature_vec.reshape(1, -1),
                np.array(self.clusters).reshape(len(self.clusters), -1),
            )
            if np.amin(distances) > self.init_dis_thres:
                self.clusters.append(feature_vec)
                self.clusters_sizes.append(num)
            else:
                nearest_idx = np.argmin(distances)
                self.clusters_sizes[nearest_idx] += num
                self.clusters[nearest_idx] += (
                    (feature_vec - self.clusters[nearest_idx])
                    * num
                    / self.clusters_sizes[nearest_idx]
                )

        else:
            distances = cosine_distance(
                feature_vec.reshape(1, -1),
                np.array(self.clusters).reshape(len(self.clusters), -1),
            )
            nearest_idx = np.argmin(distances)
            self.clusters_sizes[nearest_idx] += num
            self.clusters[nearest_idx] += (
                (feature_vec - self.clusters[nearest_idx])
                * num
                / self.clusters_sizes[nearest_idx]
            )

    def merge(self, other):
        for i, feature in enumerate(other.clusters):
            self.update(feature, other.clusters_sizes[i])

    def global_merge(self, global_feats):
        distances = cosine_distance(
            global_feats, np.array(self.clusters).reshape(len(self.clusters), -1)
        )
        for i, feat in enumerate(global_feats):
            if len(self.clusters) < self.feature_len:
                if np.amin(distances[i]) > self.init_dis_thres:
                    self.clusters.append(feat)
                    self.clusters_sizes.append(1)
                else:
                    nearest_idx = np.argmin(distances[i])
                    self.clusters[nearest_idx] = (
                        self.global_merge_weight * feat
                        + (1 - self.global_merge_weight) * self.clusters[nearest_idx]
                    )
            else:
                nearest_idx = np.argmin(distances[i])
                self.clusters[nearest_idx] = (
                    self.global_merge_weight * feat
                    + (1 - self.global_merge_weight) * self.clusters[nearest_idx]
                )

    def get_clusters_matrix(self):
        return np.array(self.clusters).reshape(len(self.clusters), -1)

    def __len__(self):
        return len(self.clusters)


# Thể hiện trạng thái của quá trình tracking
class TrackState(IntEnum):
    """
    Enumeration type for the single target track state. Newly Started tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`.

    """

    Tentative = 1
    Confirmed = 2


class Track:
    def __init__(
        self,
        id,
        cam_id,
        box,
        start_time,
        feature=None,
        num_clusters=4,
        clust_init_dis_thresh=0.1,
        budget=3,
        stable_time_thresh=5,
        rectify_length_thresh=2,
    ):
        """
        Sử dụng để theo dõi đối tượng:
            - id: ID của đối tượng
            - cam_id: ID của Camera mà đối tượng được phát hiện
            - f_queue: Một hàng đợi các đặc trưng của các đối tượng
            - f_avg: Một bộ ước tính trung bình được sử dụng để tính toán đặc trưng trung bình của đối tượng
            - f_clust: Một bộ ước tính tập cụ được sử dụng để tính toán các tập cụ đặc trưng của đối tượng
            - last_box: BBox của đối tượng được phát hiện gấn nhất
            - counts: Số lượng lần đối tượng được phát hiện
            - hits: Số lượng khung hình mà đối tượng được theo dõi
            - start_time: Thời gian bắt đầu theo dõi đối tượng
            - end_time: thời gian kết thúc theo dõi đối tượng
            - state:  Trạng thái của đối tượng (Tentative, Confirmed, Lost).
            - budget: Kích thước tối đa của hàng đợi f_queue.
            - trajectory: Một danh sách các ID camera của các camera mà đối tượng đã được phát hiện.
            - off: Một cờ cho biết liệu đối tượng đã biến mất khỏi khung hình hay chưa.
            - cross_camera_track: Một cờ cho biết liệu đối tượng đã được theo dõi giữa các camera hay chưa.
            - feats_delivery_status: Một cờ cho biết liệu đặc trưng của đối tượng đã được gửi đến máy chủ hay chưa.
            - pos_delivery_status: Một cờ cho biết liệu vị trí của đối tượng đã được gửi đến máy chủ hay chưa.
            - last_merge_dis: Khoảng cách giữa đối tượng và đối tượng khác được hợp nhất gần đây nhất.
            - stable_time_thresh: Số khung hình tối thiểu mà đối tượng phải được theo dõi để được coi là ổn định.
            - rectify_length_thresh: Số khung hình tối thiểu của đặc trưng trung bình để được coi là hợp lệ.
        """
        self.id = id
        self.cam_id = cam_id
        self.f_queue = []
        self.f_avg = AverageEstimator()  # average feature
        self.f_clust = ClusterFeature(
            num_clusters, init_dis_thres=clust_init_dis_thresh
        )  # cluster feature
        self.last_box = None
        self.counts = 0
        self.hits = 1
        self.start_time = start_time
        self.end_time = start_time
        self.state = TrackState.Tentative
        self.budget = budget
        self.trajectory = cam_id  # current camera ID
        self.off = False
        self.cross_camera_track = False
        self.feats_delivery_status = False
        self.pos_delivery_status = False
        self.last_merge_dis = 1.0
        self.stable_time_thresh = stable_time_thresh
        self.rectify_length_thresh = rectify_length_thresh

        if feature is not None:
            self.f_queue.append(feature)
        if box is not None:
            self.last_box = box

    def get_end_time(self):
        # trả về thời gian kết thúc theo dõi đối tượng
        return self.end_time

    def get_start_time(self):
        # Trả về thời gian bắt đầu theo dõi đối tượng
        return self.start_time

    def get_last_box(self):
        # Trả về bbox của đối tượng được phát hiện gần đây nhất
        return self.last_box

    def is_confirmed(self):
        # Kiểm tra xác nhận trạng thái của vật thể, trả về True nếu đối tượng đã được xác nhận, False nếu là chưa
        return self.state == TrackState.Confirmed

    def is_stable(self):
        # Trả về True nếu đối tượng được coi là ổn định, False nếu chưa.
        return (
            self.counts >= self.stable_time_thresh
            and len(self.f_avg) >= self.rectify_length_thresh
        )

    def __len__(self):
        #  Trả về số lượng khung hình mà đối tượng được theo dõi.
        return self.hits

    def get_all_features(self):
        # Trả về một danh sách tất cả các đặc trưng của đối tượng, bao gồm đặc trưng trung bình, đặc trưng tập cụ và các đặc trưng được phát hiện gần đây nhất.
        track_all_features = []
        if self.f_avg.is_valid():
            recent_features = self.f_queue
            track_all_features = track_all_features + recent_features
            avg_features = self.f_avg.get_avg()
            track_all_features.append(avg_features)
            cluster_features = self.f_clust.get_clusters_matrix()
            for i in range(cluster_features.shape[0]):
                track_all_features.append(cluster_features[i])
        else:
            recent_features = self.f_queue
            track_all_features = track_all_features + recent_features

        return track_all_features

    def enqueue_dequeue(self, feature):
        # hêm một đặc trưng mới vào hàng đợi f_queue và xóa đặc trưng cũ nhất nếu hàng đợi đã đầy.
        self.f_queue.append(feature)
        self.f_queue = self.f_queue[-self.budget :]

    def add_detection(self, box, feature, time, is_occluded):
        # Thêm một phát hiện mới vào đối tượng.
        self.last_box = box
        self.end_time = time
        self.hits += 1
        if feature is not None:
            if self.is_confirmed() and not is_occluded:
                self.f_clust.update(feature)
                self.f_avg.update(feature)
                self.enqueue_dequeue(feature)
            else:
                self.enqueue_dequeue(feature)

    def merge_continuation(self, other, dist):
        # Hợp nhất đối tượng này với một đối tượng khác có khoảng cách nhỏ hơn ngưỡng.
        self.f_queue = other.f_queue
        self.f_avg.merge(other.f_avg)
        self.f_clust.merge(other.f_clust)
        self.end_time = other.end_time
        self.hits += other.hits
        self.last_box = other.last_box
        self.last_merge_dis = dist

    def global_merge(self, track, dist):
        # Hợp nhất đối tượng này với một đối tượng khác có khoảng cách nhỏ hơn ngưỡng, bất kể đối tượng đó có thuộc camera khác hay không.
        self.f_clust.global_merge(track.f_clust)
        if self.cross_camera_track:
            if track.start_time < self.get_start_time():
                self.cam_id = track.cam_id
                self.id = track.id
                self.start_time = track.start_time
        else:
            self.cam_id = track.cam_id
            self.id = track.id
            self.start_time = track.start_time
        self.cross_camera_track = True
        self.pos_delivery_status = True
        self.last_merge_dis = dist
