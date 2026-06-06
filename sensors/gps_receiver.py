"""
sensors/gps_receiver.py
=======================
Interface for GPS Hardware and Coordinate Transformation.
"""

import math
from typing import Optional, Tuple
from dataclasses import dataclass
import time

from utils.logger import get_logger

log = get_logger(__name__)

# Earth radius in meters
R_EARTH = 6378137.0


@dataclass
class GPSReading:
    latitude: float
    longitude: float
    altitude: float = 0.0
    hdop: float = 1.0  # Horizontal dilution of precision
    timestamp: float = 0.0
    is_valid: bool = False


class GPSReceiver:
    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 9600):
        self.port = port
        self.baudrate = baudrate
        self.reference_lat = None
        self.reference_lon = None

        # In a real implementation, we would open serial port here
        # self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
        self.is_connected = False
        log.info(f"Initialized GPS receiver on {self.port}")

    def set_reference_point(self, lat: float, lon: float):
        """Set the origin (0,0) of the local coordinate system."""
        self.reference_lat = lat
        self.reference_lon = lon
        log.info(f"GPS Reference set to Lat: {lat}, Lon: {lon}")

    def get_latest_reading(self) -> GPSReading:
        """
        Reads the latest NMEA sentence from the serial port.
        (Mocked for now since hardware is unknown)
        """
        # TODO: Implement actual PySerial NMEA parsing (e.g. pynmea2)
        # Mock reading
        return GPSReading(
            latitude=0.0, longitude=0.0, timestamp=time.time(), is_valid=False
        )

    def latlon_to_local(self, lat: float, lon: float) -> Optional[Tuple[float, float]]:
        """
        Convert Latitude/Longitude to local X,Y meters from the reference point.
        Uses the Equirectangular approximation which is fine for small distances (like a farm).
        Returns (x_meters, y_meters) where:
          x is Easting
          y is Northing
        """
        if self.reference_lat is None or self.reference_lon is None:
            log.warning("Cannot convert to local coords without a reference point.")
            return None

        dlat = math.radians(lat - self.reference_lat)
        dlon = math.radians(lon - self.reference_lon)

        # Center latitude for the approximation
        lat_avg = math.radians((lat + self.reference_lat) / 2.0)

        y = R_EARTH * dlat
        x = R_EARTH * math.cos(lat_avg) * dlon

        return x, y

    def local_to_latlon(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """Convert local X,Y meters back to Latitude/Longitude."""
        if self.reference_lat is None or self.reference_lon is None:
            return None

        dlat = y / R_EARTH
        lat = self.reference_lat + math.degrees(dlat)

        lat_avg = math.radians((lat + self.reference_lat) / 2.0)
        dlon = x / (R_EARTH * math.cos(lat_avg))
        lon = self.reference_lon + math.degrees(dlon)

        return lat, lon
