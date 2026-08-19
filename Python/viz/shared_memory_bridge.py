"""
Shared-memory reader for the INR live viewer. 
Seperate from any QT stuff so we cna swap out for a different fiewer later
"""
import ctypes
import mmap
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


class VizSharedHeader(ctypes.Structure):
    # Must match VizProtocol.h's VizSharedHeader exactly: same field order,
    # same widths, packed with no padding.
    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("_PLACEHOLDER", ctypes.c_uint32),
        ("frame_counter", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("channels", ctypes.c_uint32),
        ("has_target", ctypes.c_uint32),
        ("target_width", ctypes.c_uint32),
        ("target_height", ctypes.c_uint32),
        ("running", ctypes.c_uint32),
        ("epoch", ctypes.c_uint64),
        ("cost", ctypes.c_float),
        ("learning_rate", ctypes.c_float),
    ]


MAGIC = 0x56524E49  # 'INRV' must match VizSharedHeader{}'s default in C++


@dataclass
class TrainingStats:
    epoch: int
    cost: float
    learning_rate: float
    running: bool


class SharedMemoryBridge:
    """Opens the named mappings a running C++ trainer created and lets you
    poll for the latest frame. Safe to construct before the C++ side has
    started call wait_for_producer() or retry connect() yourself"""

    def __init__(self, tag: str = "INR_Default"):
        self.tag = tag
        self._header_mmap: Optional[mmap.mmap] = None
        self._current_mmap: Optional[mmap.mmap] = None
        self._target_mmap: Optional[mmap.mmap] = None
        self._header = VizSharedHeader()

        self.width = 0
        self.height = 0
        self.channels = 0
        self.target_width = 0
        self.target_height = 0
        self.connected = False

    def connect(self) -> bool:
        """Try to open the mappings. Returns False if the C++ side hasn't
        created them yet caller should retry (see wait_for_producer)"""
        try:
            self._header_mmap = mmap.mmap(-1, ctypes.sizeof(VizSharedHeader),
                                           tagname=f"Local\\{self.tag}_Header")
        except OSError:
            return False

        self._read_header()
        if self._header.magic != MAGIC:
            self._header_mmap = None
            return False

        self.width = self._header.width
        self.height = self._header.height
        self.channels = self._header.channels
        self.target_width = self._header.target_width
        self.target_height = self._header.target_height

        current_bytes = self.width * self.height * self.channels * 4  # float32
        self._current_mmap = mmap.mmap(-1, current_bytes, tagname=f"Local\\{self.tag}_Current")

        if self._header.has_target:
            target_bytes = self.target_width * self.target_height * self.channels * 4
            self._target_mmap = mmap.mmap(-1, target_bytes, tagname=f"Local\\{self.tag}_Target")

        self.connected = True
        return True

    def wait_for_producer(self, timeout_s: float = 30.0, poll_s: float = 0.25) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.connect():
                return True
            time.sleep(poll_s)
        return False

    def _read_header(self):
        self._header_mmap.seek(0)
        buf = self._header_mmap.read(ctypes.sizeof(VizSharedHeader))
        ctypes.memmove(ctypes.byref(self._header), buf, ctypes.sizeof(VizSharedHeader))

    def read_stats(self) -> TrainingStats:
        self._read_header()
        return TrainingStats(epoch=self._header.epoch, cost=self._header.cost,
                              learning_rate=self._header.learning_rate,
                              running=bool(self._header.running))

    def last_frame_counter(self) -> int:
        self._read_header()
        return self._header.frame_counter

    def read_current_frame(self) -> Optional[np.ndarray]:
        """Seqlock read: retries a bounded number of times if a write was in
        progress mid-read. Returns an (H, W, C) float32 array, or None if the
        producer is (unusually) mid-write on every attempt caller should
        just keep showing the last good frame and try again next poll"""
        if not self.connected:
            return None
        for _ in range(8):
            self._read_header()
            c1 = self._header.frame_counter
            if c1 % 2 != 0:  # odd == write in progress
                time.sleep(0.001)
                continue
            self._current_mmap.seek(0)
            raw = self._current_mmap.read()
            self._read_header()
            c2 = self._header.frame_counter
            if c1 == c2:
                arr = np.frombuffer(raw, dtype=np.float32)
                return arr.reshape(self.height, self.width, self.channels)
        return None

    def read_target_frame(self) -> Optional[np.ndarray]:
        if not self.connected or self._target_mmap is None:
            return None
        self._target_mmap.seek(0)
        raw = self._target_mmap.read()
        arr = np.frombuffer(raw, dtype=np.float32)
        return arr.reshape(self.target_height, self.target_width, self.channels)
