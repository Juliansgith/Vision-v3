import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
# from yolox.tracker import matching # Removed as per plan
from . import matching # Use local matching module

from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, feature=None): # Added feature

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        # Initialize ReID features
        self.curr_feat = feature
        self.smooth_feat = feature # Initialize with the first feature
        self.features = deque(maxlen=10) # TODO: make maxlen configurable
        if feature is not None:
            self.features.append(feature)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        # Update features on re-activation
        self.curr_feat = new_track.curr_feat # new_track is an STrack
        if self.curr_feat is not None:
            self.features.clear()
            self.features.append(self.curr_feat)
            self.smooth_feat = self.curr_feat # Reset smoothed feature to the new current one
        else:
            # If the re-activating detection has no feature, clear existing ones
            self.features.clear()
            self.smooth_feat = None
            # self.curr_feat is already None

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

        # Update features
        self.curr_feat = new_track.curr_feat # new_track is an STrack, so it has .curr_feat
        if self.curr_feat is not None:
            self.features.append(self.curr_feat)
            
            # Update smooth_feat using EMA
            alpha = 0.9 # Smoothing factor
            if self.smooth_feat is None:
                self.smooth_feat = self.curr_feat
            else:
                # Ensure smooth_feat and curr_feat are compatible for EMA (e.g., numpy arrays)
                if isinstance(self.smooth_feat, np.ndarray) and isinstance(self.curr_feat, np.ndarray):
                    self.smooth_feat = alpha * self.smooth_feat + (1 - alpha) * self.curr_feat
                else:
                    # Fallback or error if types are not as expected
                    self.smooth_feat = self.curr_feat # Or log a warning
        # If curr_feat is None, we might not update smooth_feat or features, or clear them.
        # For now, if new detection has no feature, existing smooth_feat and features are preserved.

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, reid_config=None, frame_rate=30): # args is expected to be a namespace/object
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args # This 'args' should have track_thresh, track_buffer, match_thresh, mot20
        self.det_thresh = args.track_thresh + 0.1 # High confidence detections
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.reid_config = reid_config
        self.enable_reid = self.reid_config.enable_reid if self.reid_config and hasattr(self.reid_config, 'enable_reid') else False
        if self.enable_reid:
            print(f"[INFO] ByteTracker initialized with ReID enabled. Match Thresh: {self.reid_config.reid_match_thresh}, Fuse Weight (for ReID): {self.reid_config.reid_fuse_weight}")
        else:
            print("[INFO] ByteTracker initialized with ReID disabled.")


    def update(self, output_results, img_info, img_size, features_list=None): # Added features_list
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5: # Detections are [x1,y1,x2,y2,score]
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else: # Expected format [x1,y1,x2,y2,obj_conf,class_conf,class_pred] or similar from YOLOX
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5] # obj_conf * class_conf
            bboxes = output_results[:, :4]  # x1y1x2y2
        
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        # Create STrack objects with features for ALL raw detections
        all_detections_as_stracks = []
        for i in range(len(bboxes)): # Assuming bboxes is a list/array of bounding boxes
            feature = None
            if features_list is not None and i < len(features_list) and features_list[i] is not None:
                feature = features_list[i]
            # Convert tlbr (from bboxes) to tlwh for STrack constructor
            all_detections_as_stracks.append(STrack(STrack.tlbr_to_tlwh(bboxes[i]), scores[i], feature=feature))

        # Filter based on scores using original indices/masks
        # remain_inds, inds_second are boolean masks for the original scores/bboxes array
        remain_inds = scores > self.args.track_thresh
        # inds_low = scores > 0.1 # Not strictly needed if using all_detections_as_stracks
        # inds_high = scores < self.args.track_thresh # Not strictly needed
        # Recreate inds_second based on all_detections_as_stracks if necessary, or use original scores
        inds_second = np.logical_and(scores > 0.1, scores < self.args.track_thresh)

        detections = [s for i, s in enumerate(all_detections_as_stracks) if remain_inds[i]]
        detections_second = [s for i, s in enumerate(all_detections_as_stracks) if inds_second[i]]
        
        # No longer need separate creation for dets, scores_keep, dets_second, scores_second
        # as STrack objects in 'detections' and 'detections_second' already have this info.

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        
        # Calculate IoU distance
        iou_dists = matching.iou_distance(strack_pool, detections)
        dists = iou_dists.copy() # Initialize with IoU distances

        if self.enable_reid and len(strack_pool) > 0 and len(detections) > 0 and self.reid_config:
            tracks_with_features = [t for t in strack_pool if t.smooth_feat is not None]
            dets_with_features = [d for d in detections if d.curr_feat is not None]

            if len(tracks_with_features) > 0 and len(dets_with_features) > 0:
                track_indices = [i for i, t in enumerate(strack_pool) if t.smooth_feat is not None]
                det_indices = [i for i, d in enumerate(detections) if d.curr_feat is not None]
                
                emb_dists_subset = matching.embedding_distance(tracks_with_features, dets_with_features)
                
                fuse_weight_reid = self.reid_config.reid_fuse_weight
                reid_match_threshold = self.reid_config.reid_match_thresh

                for i, track_idx in enumerate(track_indices):
                    for j, det_idx in enumerate(det_indices):
                        iou_val = iou_dists[track_idx, det_idx]
                        emb_val = emb_dists_subset[i, j]

                        if emb_val <= reid_match_threshold:
                            dists[track_idx, det_idx] = (1.0 - fuse_weight_reid) * iou_val + fuse_weight_reid * emb_val
                        else:
                            # If ReID match is bad, make the cost very high (effectively relying on IoU or disqualifying)
                            # Using 1.0 assumes costs are normalized 0-1. A common strategy for "disqualification".
                            dists[track_idx, det_idx] = 1.0 
            # If no tracks/detections have features, dists remains iou_dists
        
        # Original ByteTrack score fusion (if not MOT20)
        if not self.args.mot20: 
            dists = matching.fuse_score(dists, detections) 
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        # detections_second is already a list of STrack objects
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        
        # Calculate IoU distance for low-score detections
        iou_dists_low = matching.iou_distance(r_tracked_stracks, detections_second)
        dists_low = iou_dists_low.copy() # Initialize with IoU distances

        if self.enable_reid and len(r_tracked_stracks) > 0 and len(detections_second) > 0 and self.reid_config:
            tracks_low_with_features = [t for t in r_tracked_stracks if t.smooth_feat is not None]
            dets_low_with_features = [d for d in detections_second if d.curr_feat is not None]

            if len(tracks_low_with_features) > 0 and len(dets_low_with_features) > 0:
                track_indices_low = [i for i, t in enumerate(r_tracked_stracks) if t.smooth_feat is not None]
                det_indices_low = [i for i, d in enumerate(detections_second) if d.curr_feat is not None]
                
                emb_dists_low_subset = matching.embedding_distance(tracks_low_with_features, dets_low_with_features)

                fuse_weight_reid = self.reid_config.reid_fuse_weight
                reid_match_threshold = self.reid_config.reid_match_thresh

                for i, track_idx in enumerate(track_indices_low):
                    for j, det_idx in enumerate(det_indices_low):
                        iou_val = iou_dists_low[track_idx, det_idx]
                        emb_val = emb_dists_low_subset[i, j]
                        
                        if emb_val <= reid_match_threshold:
                            dists_low[track_idx, det_idx] = (1.0 - fuse_weight_reid) * iou_val + fuse_weight_reid * emb_val
                        else:
                            dists_low[track_idx, det_idx] = 1.0 # Penalize bad ReID match
            # If no tracks/detections have features, dists_low remains iou_dists_low

        matches, u_track, u_detection_second = matching.linear_assignment(dists_low, thresh=0.5) # Note: thresh=0.5 is hardcoded here
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections) # Changed local_matching to matching
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections) # Changed local_matching to matching
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7) # Changed local_matching to matching
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh: # Use det_thresh (args.track_thresh + 0.1) for init
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb) # Changed local_matching to matching
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb