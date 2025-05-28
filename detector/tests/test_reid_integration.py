import unittest
import unittest.mock as mock
import numpy as np
from collections import deque
import torch
import torchvision.transforms as T
from PIL import Image # Needed for _extract_features tests

# Assuming the following imports are relative to the project root or test execution context
# Adjust these paths if necessary based on how tests are run
from ..basetrack import BaseTrack, TrackState # For STrack inheritance
from ..byte_tracker import STrack # The class we are testing
from .. import matching # For embedding_distance
from ...config import AppSettings, YOLOConfig, TrackerConfig, DeepFaceConfig, DisplayConfig, ReIDConfig # For ObjectDetector tests
from ..object_detector import ObjectDetector # For _extract_features tests

# Mock ReID Model as provided in the prompt
class MockReIDModel(torch.nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.dummy_param = torch.nn.Parameter(torch.empty(1)) 

    def forward(self, x):
        batch_size = x.shape[0]
        # Create a deterministic feature based on input mean to make it somewhat unique per input
        # This ensures that different inputs produce different (mock) features.
        mean_val = torch.mean(x, dim=(1,2,3), keepdim=True) # Mean per batch item
        feature = mean_val.expand(-1, self.feature_dim) # Expand to feature_dim
        return feature.float()


class TestSTrackFeatures(unittest.TestCase):
    def setUp(self):
        BaseTrack._count = 0 
        self.dummy_tlwh = [10, 20, 5, 15] 
        self.dummy_score = 0.9
        self.feature_dim = 256 

    def _create_dummy_feature(self, value=1.0):
        return np.full((self.feature_dim,), value, dtype=np.float32)

    def test_strack_initialization_with_feature(self):
        feature1 = self._create_dummy_feature(1.0)
        strack = STrack(self.dummy_tlwh, self.dummy_score, feature=feature1)
        self.assertIsNotNone(strack.curr_feat)
        np.testing.assert_array_equal(strack.curr_feat, feature1)
        self.assertIsNotNone(strack.smooth_feat)
        np.testing.assert_array_equal(strack.smooth_feat, feature1)
        self.assertEqual(len(strack.features), 1)
        np.testing.assert_array_equal(strack.features[0], feature1)
        self.assertEqual(strack.features.maxlen, 10)

    def test_strack_initialization_without_feature(self):
        strack = STrack(self.dummy_tlwh, self.dummy_score, feature=None)
        self.assertIsNone(strack.curr_feat)
        self.assertIsNone(strack.smooth_feat)
        self.assertEqual(len(strack.features), 0)
        self.assertEqual(strack.features.maxlen, 10)

    def test_strack_update_with_features(self):
        feature_initial = self._create_dummy_feature(0.5)
        target_strack = STrack(self.dummy_tlwh, self.dummy_score, feature=feature_initial)
        feature_new = self._create_dummy_feature(1.0)
        new_detection_strack = STrack(self.dummy_tlwh, self.dummy_score, feature=feature_new)
        target_strack.update(new_detection_strack, frame_id=1)
        np.testing.assert_array_equal(target_strack.curr_feat, feature_new)
        self.assertEqual(len(target_strack.features), 2)
        np.testing.assert_array_equal(target_strack.features[0], feature_initial)
        np.testing.assert_array_equal(target_strack.features[1], feature_new)
        expected_smooth_feat_val = 0.9 * 0.5 + (1 - 0.9) * 1.0
        expected_smooth_feat = np.full((self.feature_dim,), expected_smooth_feat_val, dtype=np.float32)
        np.testing.assert_array_almost_equal(target_strack.smooth_feat, expected_smooth_feat, decimal=6)

    def test_strack_update_no_initial_feature(self):
        target_strack = STrack(self.dummy_tlwh, self.dummy_score, feature=None)
        feature_new = self._create_dummy_feature(1.0)
        new_detection_strack = STrack(self.dummy_tlwh, self.dummy_score, feature=feature_new)
        target_strack.update(new_detection_strack, frame_id=1)
        np.testing.assert_array_equal(target_strack.curr_feat, feature_new)
        np.testing.assert_array_equal(target_strack.smooth_feat, feature_new)
        self.assertEqual(len(target_strack.features), 1)
        np.testing.assert_array_equal(target_strack.features[0], feature_new)

    def test_strack_update_new_detection_no_feature(self):
        feature_initial = self._create_dummy_feature(0.5)
        target_strack = STrack(self.dummy_tlwh, self.dummy_score, feature=feature_initial)
        new_detection_strack = STrack(self.dummy_tlwh, self.dummy_score, feature=None)
        target_strack.update(new_detection_strack, frame_id=1)
        self.assertIsNone(target_strack.curr_feat)
        np.testing.assert_array_equal(target_strack.smooth_feat, feature_initial)
        self.assertEqual(len(target_strack.features), 1)
        np.testing.assert_array_equal(target_strack.features[0], feature_initial)

    def test_strack_reactivate_with_features(self):
        feature_old = self._create_dummy_feature(0.2)
        target_strack = STrack(self.dummy_tlwh, self.dummy_score, feature=feature_old)
        target_strack.features.append(self._create_dummy_feature(0.3))
        feature_reactivate = self._create_dummy_feature(0.8)
        new_track_for_reactivate = STrack(self.dummy_tlwh, self.dummy_score, feature=feature_reactivate)
        target_strack.re_activate(new_track_for_reactivate, frame_id=2)
        np.testing.assert_array_equal(target_strack.curr_feat, feature_reactivate)
        self.assertEqual(len(target_strack.features), 1)
        np.testing.assert_array_equal(target_strack.features[0], feature_reactivate)
        np.testing.assert_array_equal(target_strack.smooth_feat, feature_reactivate)

    def test_strack_reactivate_new_track_no_feature(self):
        feature_old = self._create_dummy_feature(0.2)
        target_strack = STrack(self.dummy_tlwh, self.dummy_score, feature=feature_old)
        new_track_for_reactivate = STrack(self.dummy_tlwh, self.dummy_score, feature=None)
        target_strack.re_activate(new_track_for_reactivate, frame_id=2)
        self.assertIsNone(target_strack.curr_feat)
        self.assertEqual(len(target_strack.features), 0)
        self.assertIsNone(target_strack.smooth_feat)

class TestMatchingFunctions(unittest.TestCase):
    def setUp(self):
        BaseTrack._count = 0
        self.feature_dim = 256
        self.dummy_tlwh = [0,0,1,1]
        self.dummy_score = 0.9
    
    def _create_strack_with_feature(self, feature_val_array, feature_type="smooth"):
        # feature_val_array should be a numpy array
        strack = STrack(self.dummy_tlwh, self.dummy_score) 
        if feature_type == "smooth":
            strack.smooth_feat = feature_val_array
        elif feature_type == "current":
            strack.curr_feat = feature_val_array
        return strack

    def test_embedding_distance_basic(self):
        feat_val = np.ones(self.feature_dim, dtype=np.float32)
        track1 = self._create_strack_with_feature(feat_val, "smooth")
        det1 = self._create_strack_with_feature(feat_val, "current")
        tracks = [track1]; detections = [det1]
        cost_matrix = matching.embedding_distance(tracks, detections)
        self.assertEqual(cost_matrix.shape, (1, 1))
        np.testing.assert_almost_equal(cost_matrix[0,0], 0.0, decimal=5)

    def test_embedding_distance_known_values(self):
        feat_a_val = np.zeros(self.feature_dim, dtype=np.float32); feat_a_val[0] = 1.0
        feat_b_val = np.zeros(self.feature_dim, dtype=np.float32); feat_b_val[1] = 1.0
        feat_c_val = np.zeros(self.feature_dim, dtype=np.float32); feat_c_val[0] = -1.0
        
        track_a = self._create_strack_with_feature(feat_a_val, "smooth")
        track_b = self._create_strack_with_feature(feat_b_val, "smooth")
        det_a = self._create_strack_with_feature(feat_a_val, "current")
        det_b = self._create_strack_with_feature(feat_b_val, "current")
        det_c = self._create_strack_with_feature(feat_c_val, "current")
        
        tracks = [track_a, track_b]; detections = [det_a, det_b, det_c]
        cost_matrix = matching.embedding_distance(tracks, detections)
        self.assertEqual(cost_matrix.shape, (2, 3))
        np.testing.assert_almost_equal(cost_matrix[0,0], 0.0, decimal=5) 
        np.testing.assert_almost_equal(cost_matrix[0,1], 1.0, decimal=5) 
        np.testing.assert_almost_equal(cost_matrix[0,2], 2.0, decimal=5) 
        np.testing.assert_almost_equal(cost_matrix[1,0], 1.0, decimal=5)
        np.testing.assert_almost_equal(cost_matrix[1,1], 0.0, decimal=5)
        np.testing.assert_almost_equal(cost_matrix[1,2], 1.0, decimal=5)

    def test_embedding_distance_empty(self):
        tracks = []; detections = []
        cost_matrix = matching.embedding_distance(tracks, detections)
        self.assertEqual(cost_matrix.size, 0)
        feat_val = np.ones(self.feature_dim, dtype=np.float32)
        track1 = self._create_strack_with_feature(feat_val, "smooth")
        det1 = self._create_strack_with_feature(feat_val, "current")
        cost_matrix = matching.embedding_distance([track1], [])
        self.assertEqual(cost_matrix.size, 0)
        cost_matrix = matching.embedding_distance([], [det1])
        self.assertEqual(cost_matrix.size, 0)

class TestObjectDetectorReID(unittest.TestCase):
    def setUp(self):
        # Create a minimal AppSettings configuration for the detector
        self.config = AppSettings(
            app={'source': 'dummy_video.mp4', 'output_dir': 'test_output'}, # Dummy source
            yolo=YOLOConfig(model='yolov8n.pt', device='cpu'), # Use CPU for tests
            tracker=TrackerConfig(), # Default tracker config
            deepface=DeepFaceConfig(enable_emotion=False, enable_age_gender=False), # Disable DeepFace
            reid=ReIDConfig(enable_reid=True, reid_feature_dim=128, reid_model_path="dummy_reid.pt"), # Enable ReID
            display=DisplayConfig()
        )
        self.feature_dim = self.config.reid.reid_feature_dim

        # Mock ObjectDetector's internal methods that are not relevant to _extract_features
        # or require heavy dependencies (like YOLO model loading, video capture)
        with mock.patch.object(ObjectDetector, '_load_yolo_model', return_value=None), \
             mock.patch.object(ObjectDetector, '_init_video_capture', return_value=None), \
             mock.patch.object(ObjectDetector, '_print_opencv_instructions', return_value=None), \
             mock.patch.object(ObjectDetector, '_warmup_models', return_value=None):
            # We pass the config directly. _load_reid_model will be called in __init__
            # So, we mock torch.jit.load which is called by _load_reid_model
            with mock.patch('torch.jit.load') as mock_jit_load:
                self.mock_reid_model_instance = MockReIDModel(feature_dim=self.feature_dim)
                mock_jit_load.return_value = self.mock_reid_model_instance
                
                self.detector = ObjectDetector(config=self.config)
        
        # Ensure the mock model was loaded and preprocess is set
        self.detector.reid_model = self.mock_reid_model_instance # Explicitly set if __init__ logic is tricky
        self.detector.reid_preprocess = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.detector.config.reid.enable_reid = True # Ensure it's enabled for tests

    def test_extract_features_reid_enabled(self):
        frame_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        bboxes_xyxy = [[10, 10, 60, 110], [100, 100, 180, 280]] # Two boxes
        
        features_list = self.detector._extract_features(frame_bgr, bboxes_xyxy)
        
        self.assertEqual(len(features_list), 2)
        for feature in features_list:
            self.assertIsNotNone(feature)
            self.assertIsInstance(feature, np.ndarray)
            self.assertEqual(feature.shape, (self.feature_dim,))
            # Check L2 normalization (magnitude approx 1.0)
            norm = np.linalg.norm(feature)
            self.assertAlmostEqual(norm, 1.0, places=5, msg="Feature not L2 normalized")

    def test_extract_features_reid_disabled(self):
        self.detector.config.reid.enable_reid = False # Disable ReID
        frame_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        bboxes_xyxy = [[10, 10, 60, 110]]
        
        features_list = self.detector._extract_features(frame_bgr, bboxes_xyxy)
        
        self.assertEqual(len(features_list), 1)
        self.assertIsNone(features_list[0])

    def test_extract_features_empty_bboxes(self):
        frame_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        bboxes_xyxy = [] # Empty list
        
        features_list = self.detector._extract_features(frame_bgr, bboxes_xyxy)
        self.assertEqual(len(features_list), 0)

    def test_extract_features_invalid_bbox(self):
        frame_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        # One valid, one invalid (e.g., x1 > x2)
        bboxes_xyxy = [[10, 10, 60, 110], [100, 100, 80, 280]] 
        
        features_list = self.detector._extract_features(frame_bgr, bboxes_xyxy)
        
        self.assertEqual(len(features_list), 2)
        self.assertIsNotNone(features_list[0]) # First should be valid
        self.assertIsNone(features_list[1])    # Second should be None due to invalid crop

    def test_extract_features_l2_normalization_explicit_check(self):
        # Modify MockReIDModel to return non-normalized features to ensure _extract_features normalizes them
        class NonNormalizedMockReIDModel(torch.nn.Module):
            def __init__(self, feature_dim):
                super().__init__()
                self.feature_dim = feature_dim
                self.dummy_param = torch.nn.Parameter(torch.empty(1))
            def forward(self, x):
                batch_size = x.shape[0]
                # Return features that are definitely not L2 normalized (e.g., all 2s)
                return torch.full((batch_size, self.feature_dim), 2.0, dtype=torch.float32)

        self.detector.reid_model = NonNormalizedMockReIDModel(self.feature_dim)
        
        frame_bgr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) # Smaller frame for faster test
        bboxes_xyxy = [[10, 10, 60, 110]]
        
        features_list = self.detector._extract_features(frame_bgr, bboxes_xyxy)
        
        self.assertEqual(len(features_list), 1)
        feature = features_list[0]
        self.assertIsNotNone(feature)
        norm = np.linalg.norm(feature)
        self.assertAlmostEqual(norm, 1.0, places=5, msg="Feature not L2 normalized by _extract_features")


if __name__ == '__main__':
    unittest.main()
