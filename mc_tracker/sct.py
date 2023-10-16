from enum import IntEnum
from collections import namedtuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..utils.misc import AverageEstimator


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


class SingleCameraTracker:
    def __init__(
        self,
        cam_id,
        reid_model=None,
        initial_id=1,
        time_window=5000,
        continue_time_thresh=2,
        confirm_time_thresh=2,
        match_threshold=0.25,
        stable_time_thresh=15,
        detection_occlusion_thresh=0.5,
        iou_dist_thresh=1.4,
        n_clusters=4,
        clust_init_dis_thresh=0.1,
        budget=3,
        rectify_thresh=0.3,
        rectify_time_thresh=8,
        rectify_length_thresh=2,
        merge_thresh=0.3,
        gap=50,
    ):
        """
        Theo dõi các đối tượng trong một camera
        Một số thuộc tính của lớp này :
            - reid_model: Xét đến vieecj sử dụng mô hình ReID (Re-Identification), được sử dụng để tính toán độ tương đồng giữa các đặc trưng của đối tượng
            - _next_id: ID của đối tượng tiếp theo được tạo
            - cam_id: ID của camera mà lớp theo dõi
            - tracks: Một danh sách các đối tượng đang được theo dõi
            - time: thời gian hiện tại
            - time_window: Số khung hình tối đa mà một đối tượng có thể không được phát hiện trước khi bị xóa
            (Hiểu đơn giản thì biến này dùng để check và lưu trữ những object sau n time ko xuất hiện trong khung hình)
            - continue_time_thresh: Số khung hình tối thiểu mà một đối tượng phải được phát hiện liên tục để được coi là theo dõi ổn định
            (Xét càng ít càng tốt trong quá trình nguy hiểm )-> đảm bảo detect càng nhiều càng tốt
            - confirm_time_thresh: Số khung hình tối thiểu mà một đối tượng phải được theo dõi ổn định trước khi được coi là được xác nhận.
            (ở đoạn này nhiều ng sẽ thấy rối, nhưng chỉ cần hiểu đơn giản là detect -> ổn định -> confirm)
            - match_threshold: Ngưỡng độ tương đồng tối thiểu giữa hai đặc trưng của đối tượng để được coi là khớp.
            (Hiện tại đang xét là 0.25 -> có thể thấp hơn do yêu cầu từ môi trường và ví dụ thực tế)
            - match_threshold: Ngưỡng độ tương đồng tối thiểu giữa hai đặc trưng của đối tượng để được coi là khớp.
            (Hiện tại đang xét là 0.3)
            - n_clusters: Số lượng tập cụm được sử dụng để phân cụm các đặc trưng của đối tượng
            - clust_init_dis_thresh: Khoảng cách tối đa giữa hai đặc trưng của đối tượng để được coi là thuộc cùng một tập cụ khi khởi tạo.
            - budget: Kích thước tối đa của hàng đợi các đặc trưng của đối tượng.
            - stable_time_thresh: Số khung hình tối thiểu mà một đối tượng phải được theo dõi ổn định để được coi là ổn định.
            - detection_occlusion_thresh: Ngưỡng xác suất bị che khuất của một phát hiện để được coi là bị che khuất.
            - rectify_time_thresh: Số khung hình tối đa mà một đối tượng có thể không được phát hiện trước khi được sửa.
            - rectify_length_thresh: Số khung hình tối thiểu của đặc trưng trung bình để được coi là hợp lệ.
            - rectify_thresh: Ngưỡng độ tương đồng tối thiểu giữa đặc trưng trung bình của một đối tượng và đặc trưng trung bình của một phát hiện để được coi là khớp.
            - iou_dist_thresh: Khoảng cách IOU tối đa giữa hai bbox để được coi là khớp.
            - gap: Số khung hình tối đa mà một đối tượng có thể không được phát hiện trước khi được theo dõi lại.
        """
        self.reid_model = reid_model
        self._next_id = initial_id
        self.cam_id = cam_id
        self.tracks = []
        self.time = 0

        self.time_window = time_window
        self.continue_time_thresh = continue_time_thresh
        self.confirm_time_thresh = confirm_time_thresh
        self.match_threshold = match_threshold
        self.merge_thresh = merge_thresh
        self.n_clusters = n_clusters
        self.clust_init_dis_thresh = clust_init_dis_thresh
        self.budget = budget
        self.stable_time_thresh = stable_time_thresh
        self.detection_occlusion_thresh = detection_occlusion_thresh
        self.rectify_time_thresh = rectify_time_thresh
        self.rectify_length_thresh = rectify_length_thresh
        self.rectify_thresh = rectify_thresh
        self.iou_dist_thresh = iou_dist_thresh
        self.gap = gap

    def process(self, frame, detections, screen_edge_objects, boundary_coord):
        """
        -> Dùng để xử lý một khung hình mới và cập nhật lớp theo dõi với các phát hiện mới
            Tính đặc trưng ReID cho các phát hiện mới.
            Gán các phát hiện mới cho các đối tượng hiện có.
            Tạo các đối tượng mới cho các phát hiện không khớp với bất kỳ đối tượng hiện có nào.
            Xóa các đối tượng không được phát hiện trong một thời gian dài.
            Sửa các đối tượng không được phát hiện ổn định.
            Kiểm tra trạng thái của các đối tượng.
            Hợp nhất các đối tượng tương tự nhau.
        """
        if len(detections) > 0:
            reid_features = self._get_embeddings(frame, detections)
            assignment = self._tracks_assignment(
                detections, reid_features, screen_edge_objects
            )
            self._create_new_tracks(detections, reid_features, assignment)
        self._clear_tracks()
        self._rectify_tracks()
        self._check_tracks_state(boundary_coord)
        if self.time % self.time_window == 0:
            self._merge_tracks()
        self.time += 1

    def _tracks_assignment(self, detections, features, screen_edge_objects):
        """
        gán các phát hiện mới cho các đối tượng hiện có. Hàm này hoạt động như sau:

            - Hàm này sẽ tạo một danh sách các chỉ số của các đối tượng hiện có đang hoạt động. Các đối tượng đang hoạt động là các đối tượng được phát hiện trong một số khung hình gần đây.
            - Hàm này sẽ tính toán ma trận chi phí của các đối tượng hiện có và các phát hiện mới. Ma trận chi phí là một ma trận cho biết chi phí gán mỗi đối tượng hiện có cho mỗi phát hiện mới.
            - Hàm này sẽ sử dụng thuật toán gán để gán các phát hiện mới cho các đối tượng hiện có dựa trên ma trận chi phí.
        """
        confirmed_active_tracks_idx = []
        unconfirmed_active_tracks_idx = []
        for i, track in enumerate(self.tracks):
            if (
                self.time - track.get_end_time() <= self.continue_time_thresh
                and track.is_confirmed()
                and track.trajectory == self.cam_id
            ):
                confirmed_active_tracks_idx.append(i)
            elif (
                self.time - track.get_end_time() <= self.continue_time_thresh
                and not track.is_confirmed()
                and track.trajectory == self.cam_id
            ):
                unconfirmed_active_tracks_idx.append(i)

        # Check for the occluded and screen edge objects
        occluded_objects = np.zeros(len(detections), dtype=np.bool_)
        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(detections):
                if (
                    i != j and self._ios(det1, det2) > self.detection_occlusion_thresh
                ) or screen_edge_objects[i]:
                    occluded_objects[i] = True
                    break

        # confirmed track assignment
        confirmed_track_cost_matrix = self._confirmed_tracks_assignment_cost(
            confirmed_active_tracks_idx, detections, features
        )
        detections_assignment, remaining_track_id = self._ID_assignment(
            confirmed_active_tracks_idx,
            confirmed_track_cost_matrix,
            detections,
            features,
            self.match_threshold,
            occluded_objects,
        )

        unmatched_detectons_idx = detections_assignment == None
        if not np.any(unmatched_detectons_idx):
            return detections_assignment

        remaining_dets = []
        remaining_feats = []
        for i, stat in enumerate(unmatched_detectons_idx):
            if stat == True:
                remaining_dets.append(detections[i])
                remaining_feats.append(features[i])
        occluded_objects = np.ones(len(remaining_dets), dtype=np.bool_)
        remaining_track_id = [
            idx
            for idx in remaining_track_id
            if self.time - self.tracks[idx].get_end_time() == 1
        ]

        # unconfirmed track assignment
        unconfirmed_active_tracks_idx = (
            unconfirmed_active_tracks_idx + remaining_track_id
        )
        unconfirmed_track_cost_matrix = self._unconfirmed_tracks_assignment_cost(
            unconfirmed_active_tracks_idx, remaining_dets
        )
        unmatched_detectons_assignment, _ = self._ID_assignment(
            unconfirmed_active_tracks_idx,
            unconfirmed_track_cost_matrix,
            remaining_dets,
            remaining_feats,
            self.iou_dist_thresh,
            occluded_objects,
        )
        detections_assignment[
            detections_assignment == None
        ] = unmatched_detectons_assignment

        return detections_assignment

    def _confirmed_tracks_assignment_cost(self, tracks_idx, detections, features):
        cost_matrix = np.zeros((len(tracks_idx), len(detections)), dtype=np.float32)
        for i, idx in enumerate(tracks_idx):
            iou_dist_matrix = self.iou_distance(idx, detections)
            feat_dist_matrix = self.features_distance(idx, features)
            feat_dist_matrix[
                iou_dist_matrix > self.iou_dist_thresh
            ] = THE_BIGGEST_DISTANCE
            cost_matrix[i, :] = feat_dist_matrix
        return cost_matrix

    def _unconfirmed_tracks_assignment_cost(self, tracks_idx, detections):
        cost_matrix = np.zeros((len(tracks_idx), len(detections)), dtype=np.float32)
        for i, idx in enumerate(tracks_idx):
            iou_dist_matrix = self.iou_distance(idx, detections)
            cost_matrix[i, :] = iou_dist_matrix
        return cost_matrix

    def iou_distance(self, track_idx, detections):
        track_box = self.tracks[track_idx].get_last_box()
        cost_metrix = np.zeros((1, len(detections)), dtype=np.float32)
        for i, det in enumerate(detections):
            cost_metrix[0, i] = 1.0 - self._diou(track_box, det)
        return cost_metrix

    def features_distance(self, track_idx, features):
        track_all_features = self.tracks[track_idx].get_all_features()
        cost_metrix = cosine_distance(track_all_features, features)
        return np.amin(cost_metrix, axis=0, keepdims=True)

    def _ID_assignment(
        self,
        tracks_idx,
        cost_matrix,
        detections,
        features,
        match_threshold,
        occluded_objects,
    ):
        det_assignment = np.array([None for _ in range(cost_matrix.shape[1])])
        remaining_track_id = [idx for idx in tracks_idx]

        if cost_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < match_threshold:
                    det_assignment[j] = i

            for i, j in enumerate(det_assignment):
                if j is not None:
                    idx = tracks_idx[j]
                    remaining_track_id[j] = None
                    self.tracks[idx].add_detection(
                        detections[i], features[i], self.time, occluded_objects[i]
                    )
            remaining_track_id = list(
                filter(lambda x: x is not None, remaining_track_id)
            )
        return det_assignment, remaining_track_id

    def _create_new_tracks(self, detections, features, assignment):
        """
        Hàm _create_new_tracks() được sử dụng để tạo các đối tượng mới cho các phát hiện mới không được gán cho bất kỳ đối tượng hiện có nào. Hàm này hoạt động như sau:

            Hàm này sẽ duyệt qua danh sách các chỉ số của các đối tượng hiện có được gán cho các phát hiện mới.
            Nếu một phát hiện mới không được gán cho bất kỳ đối tượng hiện có nào (tức là chỉ số đối tượng hiện có được gán cho phát hiện đó là None), thì hàm này sẽ tạo một đối tượng mới cho phát hiện đó.
            Hàm này sẽ cập nhật danh sách các đối tượng đang được theo dõi với đối tượng mới.
        """
        for i, j in enumerate(assignment):
            if j is None:
                self.tracks.append(
                    Track(
                        self._next_id,
                        self.cam_id,
                        detections[i],
                        self.time,
                        features[i],
                        num_clusters=self.n_clusters,
                        clust_init_dis_thresh=self.clust_init_dis_thresh,
                        budget=self.budget,
                        stable_time_thresh=self.stable_time_thresh,
                        rectify_length_thresh=self.rectify_length_thresh,
                    )
                )
                self._next_id += 1

    def _clear_tracks(self):
        # Xóa các đối tượng không được phát hiện trong một thời gian dài và các đối tượng không được phát hiện ổn định.
        clear_tracks = []
        for track in self.tracks:
            # remove too short and discontinuous tracks
            if (
                self.time - track.get_end_time() > self.continue_time_thresh
                and len(track.f_avg) < self.rectify_length_thresh
            ):
                continue

            clear_tracks.append(track)
        self.tracks = clear_tracks

    def _rectify_tracks(self):
        """sửa các đối tượng không được phát hiện ổn định. Hàm này hoạt động như sau:

        Hàm này sẽ chia danh sách các đối tượng đang được theo dõi thành hai nhóm: các đối tượng đang hoạt động và các đối tượng không hoạt động. Các đối tượng đang hoạt động là các đối tượng được phát hiện trong một số khung hình gần đây. Các đối tượng không hoạt động là các đối tượng không được phát hiện trong một số khung hình gần đây.
        Hàm này sẽ tính toán ma trận khoảng cách giữa các đối tượng đang hoạt động và các đối tượng không hoạt động. Ma trận khoảng cách này cho biết khoảng cách giữa các đặc trưng ReID của các đối tượng.
        Hàm này sẽ sử dụng thuật toán gán để gán các đối tượng không hoạt động cho các đối tượng đang hoạt động dựa trên ma trận khoảng cách.
        Hàm này sẽ hợp nhất các đối tượng được gán cho nhau thành một đối tượng.
        """
        active_tracks_idx = []
        not_active_tracks_idx = []

        # divided into active and non-active track
        for i, track in enumerate(self.tracks):
            if (
                self.time - track.get_end_time() <= self.rectify_time_thresh
                and len(track.f_avg) >= self.rectify_length_thresh
                and track.trajectory == self.cam_id
            ):
                active_tracks_idx.append(i)
            elif (
                len(track.f_avg) >= self.rectify_length_thresh
                and track.trajectory == self.cam_id
            ):
                not_active_tracks_idx.append(i)

        distance_matrix = np.zeros(
            (len(active_tracks_idx), len(not_active_tracks_idx)), dtype=np.float32
        )
        for i, idx1 in enumerate(active_tracks_idx):
            for j, idx2 in enumerate(not_active_tracks_idx):
                distance_matrix[i, j] = self._get_rectification_distance(
                    self.tracks[idx1], self.tracks[idx2]
                )

        indices_rows = np.arange(distance_matrix.shape[0])
        indices_cols = np.arange(distance_matrix.shape[1])

        while len(indices_rows) > 0 and len(indices_cols) > 0:
            i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            dist = distance_matrix[i, j]
            if dist < self.rectify_thresh:
                self._concatenate_tracks(
                    active_tracks_idx[indices_rows[i]],
                    not_active_tracks_idx[indices_cols[j]],
                    dist,
                )
                distance_matrix = np.delete(distance_matrix, i, 0)
                indices_rows = np.delete(indices_rows, i)
                distance_matrix = np.delete(distance_matrix, j, 1)
                indices_cols = np.delete(indices_cols, j)
            else:
                break
        self.tracks = list(filter(lambda x: x is not None, self.tracks))

    def _check_tracks_state(self, boundary_coord):
        """
        Hàm _check_tracks_state() được sử dụng để kiểm tra trạng thái của các đối tượng đang được theo dõi. Hàm này hoạt động như sau:

        - Hàm này sẽ duyệt qua danh sách các đối tượng đang được theo dõi.
        - Đối với mỗi đối tượng, hàm này sẽ kiểm tra xem đối tượng đã được xác nhận hay chưa. Nếu đối tượng đã được xác nhận, thì hàm này sẽ kiểm tra xem đối tượng có còn trong camera hay không. Nếu đối tượng không còn trong camera, thì hàm này sẽ đặt trạng thái của đối tượng là off.
        - Hàm này cũng sẽ kiểm tra xem đối tượng đã được giao đặc điểm và vị trí hay chưa. Nếu đối tượng chưa được giao đặc điểm và vị trí, thì hàm này sẽ đặt trạng thái của đối tượng thành not_stable.
        """
        for track in self.tracks:
            if track.is_confirmed():
                if self.time - track.get_end_time() <= self.continue_time_thresh:
                    track.counts += 1
                else:
                    track.counts = 0

                # check whether track left the camera or not
                track_last_box = track.get_last_box()
                if not track.off and (
                    self.time - track.get_end_time() > 2 * self.time_window
                    or (
                        self.time - track.get_end_time() > self.time_window // 2
                        and (
                            track_last_box[0] - boundary_coord[0] <= self.gap
                            or track_last_box[1] - boundary_coord[1] <= self.gap
                            or boundary_coord[2] - track_last_box[2] <= self.gap
                            or boundary_coord[3] - track_last_box[3] <= self.gap
                        )
                    )
                ):
                    track.off = True

                # check delivery and off status
                if track.is_stable():
                    if not track.feats_delivery_status:
                        track.feats_delivery_status = True
                    if track.off:
                        track.off = False
                else:
                    if (
                        not track.pos_delivery_status
                        and track.cross_camera_track
                        and track.off
                    ):
                        track.pos_delivery_status = True
            else:
                if len(track) >= self.confirm_time_thresh:
                    track.state = TrackState.Confirmed

    def _get_rectification_distance(self, track1, track2):
        if (
            (
                track1.get_start_time() > track2.get_end_time()
                or track2.get_start_time() > track1.get_end_time()
            )
            and track1.f_avg.is_valid()
            and track2.f_avg.is_valid()
        ):
            f_complex_dist = cosine_distance(
                track1.f_clust.get_clusters_matrix(),
                track2.f_clust.get_clusters_matrix(),
            )
            f_avg_dist = cosine_distance(track1.f_avg.get_avg(), track2.f_avg.get_avg())
            return min(f_avg_dist, np.amin(f_complex_dist))
        return THE_BIGGEST_DISTANCE

    def _merge_tracks(self):
        """
        Hàm _merge_tracks() được sử dụng để hợp nhất các đối tượng đang được theo dõi có khả năng bị trùng lắp. Hàm này hoạt động như sau:

            - Hàm này sẽ tính toán ma trận khoảng cách giữa tất cả các đối tượng đang được theo dõi. Ma trận khoảng cách này cho biết khoảng cách giữa các đặc trưng ReID của các đối tượng.
            - Hàm này sẽ sử dụng thuật toán gán để gán các đối tượng cho nhau dựa trên ma trận khoảng cách.
            - Hàm này sẽ hợp nhất các đối tượng được gán cho nhau thành một đối tượng.
        """
        distance_matrix, tracks_indices = self._get_merge_distance_matrix()

        while len(tracks_indices) > 0:
            i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            dist = distance_matrix[i, j]
            if dist < self.merge_thresh:
                kept_idx = self._concatenate_tracks(
                    tracks_indices[i], tracks_indices[j], dist
                )
                deleted_idx = (
                    tracks_indices[i]
                    if kept_idx == tracks_indices[j]
                    else tracks_indices[j]
                )
                assert self.tracks[deleted_idx] is None
                if deleted_idx == tracks_indices[i]:
                    idx_to_delete = i
                    idx_to_update = j
                else:
                    assert deleted_idx == tracks_indices[j]
                    idx_to_delete = j
                    idx_to_update = i
                updated_row = self._get_updated_merge_distance_matrix_row(
                    kept_idx, deleted_idx, tracks_indices
                )
                distance_matrix[idx_to_update, :] = updated_row
                distance_matrix[:, idx_to_update] = updated_row
                distance_matrix = np.delete(distance_matrix, idx_to_delete, 0)
                distance_matrix = np.delete(distance_matrix, idx_to_delete, 1)
                tracks_indices = np.delete(tracks_indices, idx_to_delete)
            else:
                break

        self.tracks = list(filter(lambda x: x is not None, self.tracks))

    def _get_merge_distance(self, track1, track2):
        if (
            (
                track1.get_start_time()
                > track2.get_end_time()  # when two tracks' appearance time is disjoint
                or track2.get_start_time() > track1.get_end_time()
            )
            and track1.f_avg.is_valid()
            and track2.f_avg.is_valid()
        ):
            f_avg_dist = cosine_distance(track1.f_avg.get_avg(), track2.f_avg.get_avg())
            f_complex_dist = cosine_distance(
                track1.f_clust.get_clusters_matrix(),
                track2.f_clust.get_clusters_matrix(),
            )
            return min(f_avg_dist, np.amin(f_complex_dist))

        return THE_BIGGEST_DISTANCE

    def _get_merge_distance_matrix(self):
        tracks_indices = []
        for i, track in enumerate(self.tracks):
            if (
                len(track.f_avg) >= self.rectify_length_thresh
                and track.trajectory == self.cam_id
            ):
                tracks_indices.append(i)

        distance_matrix = THE_BIGGEST_DISTANCE * np.eye(
            len(tracks_indices), dtype=np.float32
        )
        for i, idx_t1 in enumerate(tracks_indices):
            for j, idx_t2 in enumerate(tracks_indices):
                if i < j:
                    distance_matrix[i, j] = self._get_merge_distance(
                        self.tracks[idx_t1], self.tracks[idx_t2]
                    )
        distance_matrix += np.transpose(distance_matrix)
        return distance_matrix, tracks_indices

    def _get_updated_merge_distance_matrix_row(
        self, update_idx, ignore_idx, alive_indices
    ):
        distance_matrix = THE_BIGGEST_DISTANCE * np.ones(
            len(alive_indices), dtype=np.float32
        )
        for i, idx in enumerate(alive_indices):
            if idx != update_idx and idx != ignore_idx:
                distance_matrix[i] = self._get_merge_distance(
                    self.tracks[update_idx], self.tracks[idx]
                )
        return distance_matrix

    def _concatenate_tracks(self, i, idx, dist):
        # merge track and check who appeared first
        if self.tracks[i].get_end_time() < self.tracks[idx].get_start_time():
            self.tracks[i].merge_continuation(self.tracks[idx], dist)
            self.tracks[idx] = None
            return i
        else:
            assert self.tracks[idx].get_end_time() < self.tracks[i].get_start_time()
            self.tracks[idx].merge_continuation(self.tracks[i], dist)
            self.tracks[i] = None
            return idx

    @staticmethod
    def _area(box):
        return max((box[2] - box[0]), 0) * max((box[3] - box[1]), 0)

    def _giou(self, b1, b2, a1=None, a2=None):
        # GIOU
        if a1 is None:
            a1 = self._area(b1)
        if a2 is None:
            a2 = self._area(b2)
        intersection = self._area(
            [max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])]
        )

        enclosing = self._area(
            [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]
        )
        u = a1 + a2 - intersection  # union area
        iou = intersection / u if u > 0 else 0
        giou = iou - (enclosing - u) / enclosing if enclosing > 0 else -1
        return giou

    def _diou(self, b1, b2, a1=None, a2=None):
        # DIOU
        if a1 is None:
            a1 = self._area(b1)
        if a2 is None:
            a2 = self._area(b2)
        intersection = self._area(
            [max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])]
        )
        u = a1 + a2 - intersection  # union area
        iou = intersection / u if u > 0 else 0

        center_x1 = (b1[2] + b1[0]) / 2
        center_y1 = (b1[3] + b1[1]) / 2
        center_x2 = (b2[2] + b2[0]) / 2
        center_y2 = (b2[3] + b2[1]) / 2
        inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2

        outer_max_x = max(b1[2], b2[2])
        outer_min_x = min(b1[0], b2[0])
        outer_max_y = max(b1[3], b2[3])
        outer_min_y = min(b1[1], b2[1])
        outer_diag = (max((outer_max_x - outer_min_x), 0)) ** 2 + (
            max((outer_max_y - outer_min_y), 0)
        ) ** 2

        diou = iou - inter_diag / outer_diag if outer_diag > 0 else -1
        return diou

    def _iou(self, b1, b2, a1=None, a2=None):
        if a1 is None:
            a1 = self._area(b1)
        if a2 is None:
            a2 = self._area(b2)
        intersection = self._area(
            [max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])]
        )

        u = a1 + a2 - intersection
        return intersection / u if u > 0 else 0

    def _ios(self, b1, b2):
        # intersection over self
        a1 = self._area(b1)
        intersection = self._area(
            [max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])]
        )
        return intersection / a1 if a1 > 0 else 0

    def _get_embeddings(self, frame, detections):
        # Tính toán đặc trưng ReID cho các phát hiện mới
        rois = []
        embeddings = []

        for rect in detections:
            left, top, right, bottom = rect
            crop = frame[top:bottom, left:right]
            rois.append(crop)

        if rois:
            embeddings = self.reid_model.forward(rois)
            embeddings = [feat.reshape(-1) for feat in embeddings]
            assert len(rois) == len(embeddings)

        return embeddings

    def get_tracked_objects(self):
        objs = []
        for track in self.tracks:
            if track.get_end_time() == self.time - 1:
                objs.append(
                    TrackedObj(
                        track.get_last_box(),
                        f"{track.cam_id}-{track.id}",
                        len(track) > self.time_window,
                    )
                )
        return objs
