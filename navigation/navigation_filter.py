"""
navigation/navigation_filter.py
===============================
Smooths steering and alignment commands.
"""
from config.config_manager import config_manager
from vision.pose_filter import PoseFilter

class NavigationFilter:
    """
    Applies deadbands and smoothing to steering angles and cross-track errors
    to prevent jittery navigation.
    """
    def __init__(self):
        alpha = config_manager.get("navigation.smoothing_alpha", 0.3)
        self.steering_filter = PoseFilter(alpha=alpha)
        self.distance_filter = PoseFilter(alpha=alpha)
        self.dead_zone_x = config_manager.get("navigation.dead_zone_x", 20.0)

    def process_steering(self, raw_steering_angle: float) -> float:
        """Smooths raw steering angle."""
        return self.steering_filter.update(raw_steering_angle)
        
    def process_alignment(self, cross_track_pixel_error: float) -> float:
        """Applies deadzone and smoothing to cross track error."""
        if abs(cross_track_pixel_error) < self.dead_zone_x:
            smoothed = self.distance_filter.update(0.0)
        else:
            smoothed = self.distance_filter.update(cross_track_pixel_error)
            
        return smoothed

    def reset(self):
        self.steering_filter.reset()
        self.distance_filter.reset()
