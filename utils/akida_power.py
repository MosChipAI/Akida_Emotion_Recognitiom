import time
import statistics

import akida


class AkidaPower:
    """
    Handles consumption measures for Akida HW Device:
     * Floor power
     * Latest measure (called after every inference)
     * Min/Max/Avg (based on last 100 measures)
    """

    _device = None
    '''Get Akida.Device handle to manage an Akida HW Device.

    :attr: `_device` is a :class: `Akida.Device`, set if a HW device was found.
    '''

    floor = None
    '''HW Device floor consumption.

    :attr: `float` to store floor consumption
    '''

    latest = None
    '''Latest consumption measure.

    :attr: `float` to store latest consumption
    '''

    min_power = None
    '''Min consumption measure on last 100 measures.

    :attr: `float` to store consumption
    '''

    max_power = None
    '''Max consumption measure on last 100 measures.

    :attr: `float` to store consumption
    '''

    avg_power = None
    '''Average consumption measure on last 100 measures.

    :attr: `float` to store average consumption
    '''

    start_fps = None
    '''Used to store start time to measure frames per second.

    :attr: 'float' time stored in ms
    '''

    fps = None
    '''Estimated frames per second measuring inference time.

    :attr: `float` to store frames per second
    '''

    def __init__(self, **kwargs):
        # Check if a HW device is available
        if akida.devices():
            # Get first device
            self._device = akida.devices()[0]
            # Enable power measure
            self._device.soc.power_measurement_enabled = True
            # Get reference power (floor)
            start = time.time()
            while self.floor is None:
                try:
                    self._get_floor_power()
                except:
                    if time.time() - start >= 1.0:
                        raise Exception(
                            "[AkidaPower]: Cannot get any power measure")
                    pass
        else:
            print("[AkidaPower] No device found. Power measurement disabled.")

    def _power(self, power_event):
        # Compute power, voltage is in µV, current in mA, power is in mW.
        return power_event.voltage * power_event.current / 1000000

    def _get_floor_power(self):
        """
        Get floor power.

        :return: (float) Floor power
        """
        if self._device:
            self.floor = self._device.soc.power_meter.floor
        return self.floor

    def get_power_stats(self):
        """
        Get power statistics. The method fetches last power measure and compute
        min/max/avg on last 100 measures.

        :return: (dict) all measures statistics in a dict ordered by var names
                 or None if no devices found.
        """
        if self._device:
            # Get last 100 measures and compute min/max/avg
            last_measures = self._device.soc.power_meter.events()
            last_power_measures = []
            last_power_measures = [
                self._power(measure) for measure in last_measures
            ]

            # Adjust floor consumption
            if self.floor > min(last_power_measures):
                self.floor = min(last_power_measures)

            self.min_power = min(last_power_measures) - self.floor
            self.max_power = max(last_power_measures) - self.floor
            self.avg_power = statistics.mean(last_power_measures) - self.floor
        return self._stats()

    def _stats(self):
        """
        Creates dict with class variables except private ones.

        :return: (dict) power statistics or None if no devices found.
        """
        if self._device == None:
            return None
        else:
            return {
                "min_power": self.min_power,
                "max_power": self.max_power,
                "avg_power": self.avg_power,
                "latest": self.latest,
                "fps": self.fps
            }

    def fps_measure(self, start_timer=False):
        """
        Start/Stop FPS measure.

        :args: (bool) start/stop timer. Default value: False
        """
        if start_timer:
            self.start_fps = time.time()
        else:
            if self.start_fps is not None:
                end = time.time()
                self.fps = 1 / (end - self.start_fps)
