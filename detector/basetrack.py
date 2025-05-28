import numpy as np
from collections import OrderedDict, deque
from typing import Optional


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    # features = [] # Replaced by self.features in derived classes
    # curr_feature = None # Replaced by self.curr_feat
    score = 0
    start_frame = 0

    # ReID related attributes
    curr_feat: Optional[np.ndarray] = None
    smooth_feat: Optional[np.ndarray] = None
    features: Optional[deque] = None # To be initialized in STrack as deque(maxlen=...)
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed