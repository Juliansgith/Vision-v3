import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def iou_batch(bb_test, bb_gt):

   # computes IUO between two bboxes in the form [x1,y1,x2,y2]
   # bb_test: detection, np.array, shape (N, 4) or (4,)
   # bb_gt: tracker prediction, np.array, shape (M, 4) or (4,)
   # returns an IoU matrix of shape (N, M)

    bb_gt = np.asarray(bb_gt)
    bb_test = np.asarray(bb_test)

    if bb_test.ndim == 1:
        bb_test = bb_test[np.newaxis, :]
    if bb_gt.ndim == 1:
        bb_gt = bb_gt[np.newaxis, :]

    xx1 = np.maximum(bb_test[:, np.newaxis, 0], bb_gt[np.newaxis, :, 0])
    yy1 = np.maximum(bb_test[:, np.newaxis, 1], bb_gt[np.newaxis, :, 1])
    xx2 = np.minimum(bb_test[:, np.newaxis, 2], bb_gt[np.newaxis, :, 2])
    yy2 = np.minimum(bb_test[:, np.newaxis, 3], bb_gt[np.newaxis, :, 3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    
    area_test = (bb_test[:, np.newaxis, 2] - bb_test[:, np.newaxis, 0]) * \
                (bb_test[:, np.newaxis, 3] - bb_test[:, np.newaxis, 1])
    area_gt = (bb_gt[np.newaxis, :, 2] - bb_gt[np.newaxis, :, 0]) * \
              (bb_gt[np.newaxis, :, 3] - bb_gt[np.newaxis, :, 1])
              
    union = area_test + area_gt - wh
    iou = wh / (union + 1e-7) 
    return iou


def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    
    r = w / float(h) if h != 0 else (w / 0.00001 if w!=0 else 0.00001) 
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    s = x[2]
    r = x[3]

    if s <= 0 or r <= 0:
        w = 1 
        h = 1
    else:
        w = np.sqrt(s * r)
        h = s / w

    if(score==None):
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1,5))

class KalmanBoxTracker(object):
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. 
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.last_observation_score = bbox[4] if len(bbox) == 5 else 1.0 # Store score of last observation

    def update(self, bbox):
        self.time_since_update = 0
        self.history = [] 
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.last_observation_score = bbox[4] if len(bbox) == 5 else 1.0

    def predict(self):
        if self.kf.x[2] + self.kf.x[6] <= 0: 
             self.kf.x[6] *= 0.0 
        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections, trackers_predictions, iou_threshold = 0.3):
    if(len(trackers_predictions)==0): 
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0),dtype=int) 
    if(len(detections)==0): # No detections
        return np.empty((0,2),dtype=int), np.empty((0),dtype=int), np.arange(len(trackers_predictions))
    
    det_boxes = detections[:, :4]

    iou_matrix = iou_batch(det_boxes, trackers_predictions) 

    cost_matrix = 1.0 - iou_matrix
    
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ValueError as e:
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.arange(len(trackers_predictions))


    matched_indices_list = []
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matched_indices_list.append([r, c])
    
    if not matched_indices_list:
        matches_output_np = np.empty((0,2),dtype=int)
    else:
        matches_output_np = np.array(matched_indices_list)

    unmatched_detections = []
    detected_indices_in_matches = set(matches_output_np[:,0]) if matches_output_np.size > 0 else set()
    for d in range(len(detections)):
        if d not in detected_indices_in_matches:
            unmatched_detections.append(d)
    
    unmatched_trackers = []
    tracker_indices_in_matches = set(matches_output_np[:,1]) if matches_output_np.size > 0 else set()
    for t in range(len(trackers_predictions)):
        if t not in tracker_indices_in_matches:
            unmatched_trackers.append(t)

    return matches_output_np, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0 

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        
        predicted_tracker_boxes = []
        indices_of_trackers_to_pop = [] 
        
        for t_idx, trk_obj in enumerate(self.trackers):
            pos = trk_obj.predict()[0] 
            if np.any(np.isnan(pos)):
                indices_of_trackers_to_pop.append(t_idx)
                continue
            predicted_tracker_boxes.append(pos)
        
        for t_idx in sorted(indices_of_trackers_to_pop, reverse=True):
            self.trackers.pop(t_idx)
        
        trks_np_predictions = np.array(predicted_tracker_boxes) 

        matched, unmatched_dets, unmatched_trks_indices = \
            associate_detections_to_trackers(dets, trks_np_predictions, self.iou_threshold)

        for m in matched: 
            detection_data = dets[m[0], :]
            tracker_object_index = m[1] 
            if tracker_object_index < len(self.trackers):
                 self.trackers[tracker_object_index].update(detection_data)

        for i in unmatched_dets: 
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)

        active_tracks_to_return = []
        final_trackers_list = []

        for trk in self.trackers:
           
            if trk.time_since_update > self.max_age: 
                continue 
            current_pos_and_score = trk.get_state()[0]            
            
            if (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                track_data_to_return = np.concatenate((current_pos_and_score, [trk.id + 1])).reshape(1,-1) 
                active_tracks_to_return.append(track_data_to_return)
            
            final_trackers_list.append(trk)

        self.trackers = final_trackers_list 

        if len(active_tracks_to_return) > 0:
            return np.concatenate(active_tracks_to_return)
        return np.empty((0,5))