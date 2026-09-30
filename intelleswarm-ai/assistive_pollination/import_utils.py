"""
Import utilities for assistive_pollination system.

Provides mock initialization for integration testing.
"""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

def initialize_assistive_pollination():
    """
    Initialize the assistive_pollination system for testing.

    This creates mock dependencies and sets up the import environment
    so that integration tests can run without requiring all the external
    dependencies of the assistive_pollination system.
    """

    # Add current directory to path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    # Mock common dependencies that might not be available
    try:
        import cv2
        # Even if cv2 is available, we might need to mock imread for non-existent files
        original_imread = cv2.imread
        def mock_imread(filename, *args, **kwargs):
            result = original_imread(filename, *args, **kwargs)
            if result is None:
                # Return a mock image for non-existent files
                import numpy as np
                return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            return result
        cv2.imread = mock_imread
    except ImportError:
        # Create mock cv2 module
        cv2_mock = MagicMock()
        def mock_imread_func(*args, **kwargs):
            import numpy as np
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2_mock.imread = mock_imread_func
        cv2_mock.imread.return_value = mock_imread_func()
        cv2_mock.imread.return_value.shape = (480, 640, 3)  # Mock image shape
        sys.modules['cv2'] = cv2_mock

    try:
        import rospy
    except ImportError:
        # Create mock rospy
        rospy_mock = MagicMock()
        sys.modules['rospy'] = rospy_mock

    try:
        import rospkg
    except ImportError:
        # Create mock rospkg
        rospkg_mock = MagicMock()
        sys.modules['rospkg'] = rospkg_mock

    try:
        import actionlib
    except ImportError:
        # Create mock actionlib
        actionlib_mock = MagicMock()
        sys.modules['actionlib'] = actionlib_mock

    # Initialize other common mocks for agricultural system
    try:
        from unittest.mock import patch
        # These patches will help tests that try to use real hardware
        if 'models' not in globals():
            global models
            models = MagicMock()
    except ImportError:
        pass

    return True