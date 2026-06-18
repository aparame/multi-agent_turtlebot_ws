import os
import sys
import unittest

import numpy as np

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

from cv_localization.image_utils import crop_resize_workspace_frame


class TestCVMppiDirectGuiHelpers(unittest.TestCase):
    def test_crop_resize_workspace_frame_outputs_requested_square(self):
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                frame[y, x] = (x, y, x + y)
        corners = np.array([
            [2, 2],
            [13, 2],
            [13, 9],
            [2, 9],
        ], dtype=np.float32)

        cropped = crop_resize_workspace_frame(frame, corners, 224)

        self.assertEqual(cropped.shape, (224, 224, 3))
        self.assertGreater(float(cropped.mean()), 0.0)


if __name__ == "__main__":
    unittest.main()
