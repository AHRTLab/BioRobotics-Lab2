"""
bioradio.py - Pure Python/pyserial interface for the GLNeuroTech BioRadio device.

Replaces the .NET BioRadioSDK with a standalone Python implementation.
Communicates via serial (COM) ports using the BioRadio's custom binary protocol.

Requirements:
    pip install pyserial

Usage:
    from src.bioradio import BioRadio, scan_for_bioradio

    # Auto-scan for device
    ports = scan_for_bioradio()

    # Connect with explicit ports
    radio = BioRadio(port_in="COM9", port_out="COM10")
    radio.connect()
    config = radio.get_configuration()
    radio.start_acquisition()

    # Read data
    while True:
        data = radio.read_data(timeout=1.0)
        if data:
            print(data)

    radio.stop_acquisition()
    radio.disconnect()

Author:  BioRobotics Lab (auto-generated from BioRadioSDK analysis)
License: Educational use
"""

import struct
import time
import threading
import logging
from enum import IntEnum, IntFlag
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from collections import deque
import math

import serial
import serial.tools.list_ports

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("bioradio")

# ---------------------------------------------------------------------------
# Protocol Constants
# ---------------------------------------------------------------------------
SYNC_BYTE = 0xF0
BAUD_RATE = 460800
DEFAULT_TIMEOUT_MS = 5500
COMMAND_TIMEOUT_MS = 1000
MAX_RETRIES = 5
WATCHDOG_TIMEOUT_S = 5.0

# Unlock key: ASCII 'C','M','D','I'
UNLOCK_KEY = bytes([0x43, 0x4D, 0x44, 0x49])

VALID_SAMPLE_RATES = [250, 500, 1000, 2000, 4000, 8000, 16000]
VALID_BIT_RESOLUTIONS = [12, 16, 24]


# ---------------------------------------------------------------------------
# Enums  (match the .NET SDK exactly)
# ---------------------------------------------------------------------------
class DeviceCommand(IntEnum):
    """Command bytes (upper nibble of header byte)."""
    NegativeAck      = 0x00
    SetMode          = 0x20
    GetMode          = 0x30
    SetParam         = 0x40
    GetParam         = 0x50
    SetState         = 0x60
    PacketLength     = 0x70
    WriteEEProm      = 0x80
    ReadEEProm       = 0x90
    TransmitData     = 0xA0
    ReceiveData      = 0xB0
    MiscData         = 0xC0
    PassThroughCmd   = 0xD0
    GetGlobal        = 0xF0


class ParamId(IntEnum):
    """Sub-command byte for Get/SetParam (Data[0])."""
    CommonDAQ        = 0x01
    ChannelConfig    = 0x02
    DeviceTime       = 0x03
    BatteryStatus    = 0x04
    FirmwareVersion  = 0x05
    UnlockDevice     = 0x0E
    LockDevice       = 0x0F


class AcquisitionState(IntEnum):
    Start = 0x02
    Stop  = 0x03


class ChannelTypeCode(IntEnum):
    BioPotential = 0
    EventMarker  = 1
    Mems         = 2
    Auxiliary    = 3
    PulseOx      = 4
    NotConnected = 255


class ConfigFlags(IntFlag):
    ConnCheck   = 0x01
    DrvGround   = 0x02
    SingleEnded = 0x04


class BioPotentialMode(IntEnum):
    Normal     = 0
    GSR        = 1
    TestSignal = 2
    RIP        = 3


class CouplingType(IntEnum):
    DC = 0
    AC = 1


class StatusCode(IntEnum):
    RSSI            = 0
    SDCardRemaining = 1
    BatteryVoltage  = 2
    BatteryCurrent  = 3
    BatteryCharge   = 4
    DCDCTemp        = 5
    AmbientTemp     = 6
    ErrorCode       = 7


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class Packet:
    """A parsed BioRadio protocol packet."""
    command: DeviceCommand = DeviceCommand.NegativeAck
    data: bytes = b""
    is_response: bool = False

    @property
    def length(self) -> int:
        return len(self.data)


@dataclass
class ChannelConfig:
    """Configuration for a single channel."""
    channel_index: int = 0
    type_code: ChannelTypeCode = ChannelTypeCode.NotConnected
    name: str = ""
    preset_code: int = 0
    enabled: bool = False
    connected: bool = False
    saved: bool = True
    streamed: bool = True
    # BioPotential-specific
    gain: int = 0
    operation_mode: BioPotentialMode = BioPotentialMode.Normal
    coupling: CouplingType = CouplingType.DC
    bit_resolution: int = 12

    @classmethod
    def from_bytes(cls, raw: bytes) -> "ChannelConfig":
        """Parse channel config from device response bytes (skip param ID byte)."""
        if len(raw) < 35:
            raise ValueError(f"Channel config too short: {len(raw)} bytes")

        ch = cls()
        ch.channel_index = raw[0]
        ch.type_code = ChannelTypeCode(raw[1])

        # Name: bytes 2-31 (30 chars ASCII)
        name_bytes = raw[2:32]
        null_idx = name_bytes.find(0)
        ch.name = name_bytes[:null_idx if null_idx >= 0 else 30].decode("ascii", errors="replace")

        # Preset code: big-endian uint16
        ch.preset_code = (raw[32] << 8) | raw[33]

        # Flags byte
        flags = raw[34]
        ch.enabled   = bool(flags & 0x80)
        ch.connected = bool(flags & 0x40)
        ch.saved     = bool(flags & 0x20)
        ch.streamed  = bool(flags & 0x10)

        # BioPotential-specific fields (bytes 35-38)
        if ch.type_code == ChannelTypeCode.BioPotential and len(raw) >= 39:
            ch.gain = raw[35]
            ch.operation_mode = BioPotentialMode(raw[36])
            ch.coupling = CouplingType(raw[37])
            ch.bit_resolution = raw[38]

        return ch

    def __repr__(self):
        status = "ON " if self.enabled else "OFF"
        return (f"Ch{self.channel_index:2d} [{status}] {self.type_code.name:14s} "
                f"'{self.name}' gain={self.gain} {self.bit_resolution}bit "
                f"{self.coupling.name} {self.operation_mode.name}")


@dataclass
class DeviceConfig:
    """Global BioRadio configuration."""
    name: str = ""
    config_flags: ConfigFlags = ConfigFlags(0)
    frequency_multiplier: int = 1
    channels: List[ChannelConfig] = field(default_factory=list)

    @property
    def sample_rate(self) -> int:
        return self.frequency_multiplier * 250

    @sample_rate.setter
    def sample_rate(self, value: int):
        if value not in VALID_SAMPLE_RATES:
            raise ValueError(f"Invalid sample rate {value}. Valid: {VALID_SAMPLE_RATES}")
        self.frequency_multiplier = value // 250

    @property
    def is_single_ended(self) -> bool:
        return bool(self.config_flags & ConfigFlags.SingleEnded)

    @property
    def max_biopotential_channels(self) -> int:
        return 8 if self.is_single_ended else 4

    @classmethod
    def from_bytes(cls, raw: bytes) -> "DeviceConfig":
        """Parse global config from GetParam 0x01 response (skip param ID byte)."""
        cfg = cls()
        # Name: 16 bytes ASCII
        name_bytes = raw[0:16]
        null_idx = name_bytes.find(0)
        cfg.name = name_bytes[:null_idx if null_idx >= 0 else 16].decode("ascii", errors="replace")
        cfg.config_flags = ConfigFlags(raw[16])
        cfg.frequency_multiplier = raw[17]
        return cfg

    @property
    def biopotential_channels(self) -> List[ChannelConfig]:
        return [c for c in self.channels
                if c.type_code == ChannelTypeCode.BioPotential]

    @property
    def enabled_biopotential(self) -> List[ChannelConfig]:
        max_ch = self.max_biopotential_channels
        return [c for c in self.biopotential_channels
                if c.enabled and c.channel_index <= max_ch]

    @property
    def enabled_auxiliary(self) -> List[ChannelConfig]:
        return [c for c in self.channels
                if c.type_code == ChannelTypeCode.Auxiliary and c.enabled]

    @property
    def enabled_pulseox(self) -> List[ChannelConfig]:
        return [c for c in self.channels
                if c.type_code == ChannelTypeCode.PulseOx and c.enabled]

    @property
    def mems_enabled(self) -> bool:
        return any(c.type_code == ChannelTypeCode.Mems and c.enabled
                   for c in self.channels)

    def __repr__(self):
        return (f"BioRadio '{self.name}' | {self.sample_rate}Hz | "
                f"{'SE' if self.is_single_ended else 'DIFF'} | "
                f"{len(self.enabled_biopotential)} BioPot channels active")


@dataclass
class BatteryInfo:
    voltage: float = 0.0

    @property
    def percentage(self) -> float:
        """Rough battery percentage (3.0V=empty, 4.2V=full for Li-ion)."""
        return max(0.0, min(100.0, (self.voltage - 3.0) / 1.2 * 100))


@dataclass
class DataSample:
    """A single parsed data packet's worth of samples."""
    packet_id: int = 0
    timestamp: float = 0.0
    biopotential: Dict[int, List[int]] = field(default_factory=dict)   # ch_index -> [samples]
    auxiliary: Dict[int, int] = field(default_factory=dict)             # ch_index -> value
    pulseox: Dict[int, dict] = field(default_factory=dict)              # ch_index -> {hr, spo2, ppg}
    battery_voltage: float = 0.0
    event_marker: bool = False


# ---------------------------------------------------------------------------
# COM Port Scanner
# ---------------------------------------------------------------------------
def scan_for_bioradio(verbose: bool = True) -> List[str]:
    """
    Scan all COM ports and return those that might be a BioRadio.

    The BioRadio typically creates a paired set of COM ports via its
    Bluetooth or USB-serial bridge. Look for FTDI or "Standard Serial"
    ports at high baud rates.

    Returns:
        List of COM port names (e.g. ['COM9', 'COM10'])
    """
    candidates = []
    ports = serial.tools.list_ports.comports()
    if verbose:
        print(f"\n{'='*60}")
        print(f"  BioRadio COM Port Scanner")
        print(f"{'='*60}")
        print(f"  Found {len(ports)} COM port(s):\n")
    for p in sorted(ports, key=lambda x: x.device):
        desc = p.description or ""
        mfr = p.manufacturer or ""
        hwid = p.hwid or ""
        is_candidate = any(kw in desc.lower() + mfr.lower() + hwid.lower()
                          for kw in ["ftdi", "serial", "bioradio", "usb",
                                     "bluetooth", "standard"])
        tag = " <-- possible BioRadio" if is_candidate else ""
        if verbose:
            print(f"  {p.device:8s}  {desc:35s}  {mfr}{tag}")
        if is_candidate:
            candidates.append(p.device)

    if verbose:
        print(f"\n  Candidates: {candidates if candidates else 'None found'}")
        print(f"{'='*60}\n")
    return candidates


def probe_bioradio_port(port_name: str, timeout: float = 2.0) -> bool:
    """
    Attempt to open a port and send a GetGlobal command to see if a
    BioRadio responds. Returns True if we get a valid response.
    """
    try:
        ser = serial.Serial(
            port=port_name,
            baudrate=BAUD_RATE,
            timeout=timeout,
            write_timeout=timeout,
        )
        # Send GetGlobal firmware version: F0 F1 00
        # Header = 0xF0 (GetGlobal) | 0x01 (len=1) = 0xF1
        ser.write(bytes([SYNC_BYTE, 0xF1, 0x00]))
        ser.flush()
        time.sleep(0.5)

        # Try to read response
        response = ser.read(ser.in_waiting or 64)
        ser.close()

        # Check if we got a sync byte back
        return SYNC_BYTE in response
    except (serial.SerialException, OSError):
        return False


# ---------------------------------------------------------------------------
# Packet Builder / Parser
# ---------------------------------------------------------------------------
def build_packet(command: DeviceCommand, data: bytes = b"",
                 use_checksum: bool = False) -> bytes:
    """
    Build a raw BioRadio protocol packet.

    Frame: [0xF0] [header] [length?] [data...] [checksum?]

    The SDK sends commands WITHOUT checksum (usesChecksum is temporarily
    set to False during SendDirectCommand), but data packets from the
    device DO include checksums.
    """
    header = int(command)
    data_len = len(data)

    if data_len < 6:
        header |= data_len
        pkt = bytes([SYNC_BYTE, header]) + data
    else:
        header |= 0x06
        pkt = bytes([SYNC_BYTE, header, data_len]) + data

    if use_checksum:
        csum = SYNC_BYTE + (header & 0xFF)
        if data_len >= 6:
            csum += data_len
        for b in data:
            csum += b
        csum &= 0xFFFF
        pkt += struct.pack(">H", csum)

    return pkt


class PacketParser:
    """
    State machine that mirrors HardwareLinkHandler.ProcessData().
    Feeds raw bytes in, emits parsed Packet objects via callback.
    """

    class State(IntEnum):
        SYNC     = 0
        HEADER   = 1
        LENGTH   = 3
        DATA     = 4
        CHKSUM1  = 5
        CHKSUM2  = 6
        LONG_HI  = 7
        LONG_LO  = 8

    def __init__(self, on_packet: Callable[[Packet], None],
                 on_bad_packet: Optional[Callable[[str], None]] = None,
                 uses_checksum: bool = True):
        self.on_packet = on_packet
        self.on_bad_packet = on_bad_packet or (lambda msg: logger.warning(f"Bad packet: {msg}"))
        self.uses_checksum = uses_checksum
        self._reset()

    def _reset(self):
        self._state = self.State.SYNC
        self._current = Packet()
        self._data_buf = bytearray()
        self._data_expected = 0
        self._calc_checksum = 0
        self._recv_checksum = 0
        self._predetermined_length = 0
        self._pending_length = 0  # temp for PacketLength command

    def feed(self, raw: bytes):
        """Feed raw bytes from the serial port into the parser."""
        for b in raw:
            self._process_byte(b)

    def _process_byte(self, b: int):
        st = self._state

        if st == self.State.SYNC:
            if b == SYNC_BYTE:
                self._state = self.State.HEADER
                self._data_buf = bytearray()
                self._calc_checksum = b
                self._current = Packet()
            return

        if st == self.State.HEADER:
            self._calc_checksum += b
            cmd_nibble = b & 0xF0
            length_nibble = b & 0x07
            is_response = bool(b & 0x08)

            try:
                self._current.command = DeviceCommand(cmd_nibble)
            except ValueError:
                self._on_bad("Unknown command nibble 0x{:02X}".format(cmd_nibble))
                self._state = self.State.SYNC
                return

            self._current.is_response = is_response

            # PacketLength command
            if self._current.command == DeviceCommand.PacketLength:
                self._state = self.State.LONG_HI
                return

            if length_nibble <= 5:
                if length_nibble == 0:
                    # Zero-length response
                    if is_response:
                        self._current.data = b""
                        self._emit_response()
                    self._state = self.State.SYNC
                    return

                actual_len = length_nibble
                if self.uses_checksum:
                    actual_len -= 2
                if actual_len < 0:
                    self._on_bad("Negative data length after checksum adjustment")
                    self._state = self.State.SYNC
                    return

                self._data_expected = actual_len
                self._state = self.State.DATA
                return

            if length_nibble == 6:
                self._state = self.State.LENGTH
                return

            # length_nibble == 7: predetermined length packet
            if self.uses_checksum and self._predetermined_length >= 2:
                self._data_expected = self._predetermined_length - 2
            else:
                self._data_expected = self._predetermined_length
            self._state = self.State.DATA
            return

        if st == self.State.LENGTH:
            self._calc_checksum += b
            actual_len = b
            if self.uses_checksum:
                actual_len -= 2
            if actual_len <= 0:
                self._state = self.State.SYNC
                return
            self._data_expected = actual_len
            self._state = self.State.DATA
            return

        if st == self.State.DATA:
            self._data_buf.append(b)
            self._calc_checksum += b
            if len(self._data_buf) >= self._data_expected:
                self._current.data = bytes(self._data_buf)
                if self._current.is_response:
                    self._emit_response()
                    self._state = self.State.CHKSUM1 if self.uses_checksum else self.State.SYNC
                    # For responses, the SDK doesn't check checksum - goes straight to SYNC
                    # Actually it does go to checksum if usesChecksum, but for direct
                    # commands usesChecksum is false. Let's follow the actual logic:
                    if not self.uses_checksum:
                        self._state = self.State.SYNC
                    else:
                        self._state = self.State.CHKSUM1
                elif not self.uses_checksum:
                    self._emit_data()
                    self._state = self.State.SYNC
                else:
                    self._state = self.State.CHKSUM1
            return

        if st == self.State.CHKSUM1:
            self._recv_checksum = b << 8
            self._state = self.State.CHKSUM2
            return

        if st == self.State.CHKSUM2:
            self._recv_checksum |= b
            calc = self._calc_checksum & 0xFFFF
            if calc == self._recv_checksum:
                # Valid checksum - update predetermined length if this was a PacketLength
                if self._pending_length > 0:
                    self._predetermined_length = self._pending_length
                    self._pending_length = 0
                else:
                    self._emit_data()
            else:
                self._on_bad(f"Checksum mismatch: calc=0x{calc:04X} recv=0x{self._recv_checksum:04X}")
            self._state = self.State.SYNC
            return

        if st == self.State.LONG_HI:
            self._calc_checksum += b
            self._pending_length = b << 8
            self._state = self.State.LONG_LO
            return

        if st == self.State.LONG_LO:
            self._calc_checksum += b
            self._pending_length |= b
            if self.uses_checksum:
                self._state = self.State.CHKSUM1
            else:
                self._predetermined_length = self._pending_length
                self._pending_length = 0
                self._state = self.State.SYNC
            return

    def _emit_response(self):
        """Deliver a response packet (command reply)."""
        self.on_packet(Packet(
            command=self._current.command,
            data=bytes(self._data_buf) if self._data_buf else self._current.data,
            is_response=True
        ))

    def _emit_data(self):
        """Deliver a data packet (streaming or async)."""
        self.on_packet(Packet(
            command=self._current.command,
            data=bytes(self._data_buf) if self._data_buf else self._current.data,
            is_response=False
        ))

    def _on_bad(self, msg: str):
        self.on_bad_packet(msg)
        self._state = self.State.SYNC


# ---------------------------------------------------------------------------
# BioPotential Bit Extraction (mirrors ExtractBioPotentialValueFromByteArray)
# ---------------------------------------------------------------------------
def extract_biopotential_value(source: bytes, byte_pos: int,
                                start_bit: int, bit_length: int) -> int:
    """
    Extract a sign-extended biopotential sample from a bit-packed byte array.

    Mirrors the C# ExtractBioPotentialValueFromByteArray exactly.

    Args:
        source: Raw packet data bytes
        byte_pos: Starting byte position
        start_bit: Starting bit offset within that byte (0 or 4)
        bit_length: Resolution in bits (12, 16, or 24)

    Returns:
        Sign-extended integer value
    """
    if start_bit not in (0, 4):
        raise ValueError(f"start_bit must be 0 or 4, got {start_bit}")
    if bit_length not in (12, 16, 24):
        raise ValueError(f"bit_length must be 12/16/24, got {bit_length}")
    if byte_pos >= len(source):
        raise IndexError(f"byte_pos {byte_pos} out of range (len={len(source)})")

    is_nibble = (start_bit == 4)
    mask = 0x0F if is_nibble else 0xFF

    if bit_length == 12:
        raw = ((source[byte_pos] & mask) << (4 + start_bit)) | \
              (source[byte_pos + 1] >> (4 - start_bit))
    elif bit_length == 16:
        raw = ((source[byte_pos] & mask) << (8 + start_bit)) | \
              (source[byte_pos + 1] << start_bit)
        if is_nibble:
            raw |= (source[byte_pos + 2] >> start_bit)
    else:  # 24
        raw = ((source[byte_pos] & mask) << (16 + start_bit)) | \
              (source[byte_pos + 1] << (8 + start_bit)) | \
              (source[byte_pos + 2] << start_bit)
        if is_nibble:
            raw |= (source[byte_pos + 3] >> start_bit)

    # Sign extension
    shift = 32 - bit_length
    raw = (raw << shift) & 0xFFFFFFFF
    # Arithmetic right shift (Python handles this natively for signed ints)
    if raw & 0x80000000:
        raw = raw - 0x100000000
    raw >>= shift
    return raw


# ---------------------------------------------------------------------------
# Main BioRadio Class
# ---------------------------------------------------------------------------
class BioRadio:
    """
    Pure Python interface to the GLNeuroTech BioRadio.

    Communicates via serial ports using pyserial. Supports the full
    device protocol: connect, configure, acquire, parse data.

    Args:
        port_in:  COM port for incoming data (device -> PC).
                  On a dual-port setup, this receives streaming data.
        port_out: COM port for outgoing commands (PC -> device).
                  If None, uses port_in for both directions (single-port mode).
        baud:     Baud rate (default 460800)
    """

    def __init__(self, port_in: str = "COM9", port_out: Optional[str] = "COM10",
                 baud: int = BAUD_RATE):
        self.port_in_name = port_in
        self.port_out_name = port_out or port_in
        self.baud = baud

        self._ser_in: Optional[serial.Serial] = None
        self._ser_out: Optional[serial.Serial] = None

        self.config: Optional[DeviceConfig] = None
        self.battery: BatteryInfo = BatteryInfo()
        self.firmware_version: str = ""
        self.hardware_version: str = ""
        self.device_name: str = ""

        self._is_connected = False
        self._is_acquiring = False
        self._is_locked = False

        # Packet parser
        self._response_event = threading.Event()
        self._last_response: Optional[Packet] = None
        self._response_lock = threading.Lock()

        # Data streaming
        self._data_queue: deque = deque(maxlen=1000)
        self._data_callbacks: List[Callable[[DataSample], None]] = []
        self._listener_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._packet_count: int = 0
        self._first_packet_id: Optional[int] = None
        self._last_packet_count: int = 0
        self._dropped_packets: int = 0
        self._total_packets: int = 0

        # Watchdog
        self._watchdog_timer: Optional[threading.Timer] = None
        self._watchdog_enabled = False

        # Packet parser (for incoming data stream with checksums)
        self._parser = PacketParser(
            on_packet=self._on_packet_received,
            on_bad_packet=lambda msg: logger.debug(f"Bad packet: {msg}"),
            uses_checksum=True
        )

        # Parser for command responses (no checksum on sent commands)
        self._cmd_parser = PacketParser(
            on_packet=self._on_response_received,
            on_bad_packet=lambda msg: logger.debug(f"Bad cmd response: {msg}"),
            uses_checksum=False
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_acquiring(self) -> bool:
        return self._is_acquiring

    @property
    def dropped_packets(self) -> int:
        return self._dropped_packets

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect(self):
        """Open serial ports and initialize the device."""
        if self._is_connected:
            logger.info("Already connected")
            return

        logger.info(f"Connecting: IN={self.port_in_name}, OUT={self.port_out_name}")

        try:
            self._ser_out = serial.Serial(
                port=self.port_out_name,
                baudrate=self.baud,
                timeout=0.5,
                write_timeout=0.5,
            )
        except serial.SerialException as e:
            raise ConnectionError(f"Cannot open output port {self.port_out_name}: {e}")

        if self.port_in_name != self.port_out_name:
            try:
                self._ser_in = serial.Serial(
                    port=self.port_in_name,
                    baudrate=self.baud,
                    timeout=0.5,
                    write_timeout=0.5,
                )
            except serial.SerialException as e:
                self._ser_out.close()
                raise ConnectionError(f"Cannot open input port {self.port_in_name}: {e}")
        else:
            self._ser_in = self._ser_out

        # Get firmware version
        self._get_firmware_version()

        # Start listener thread
        self._start_listener()

        # Get device ID
        self._get_device_id()

        self._is_connected = True
        logger.info(f"Connected to BioRadio '{self.device_name}' "
                     f"(FW: {self.firmware_version}, HW: {self.hardware_version})")

    def disconnect(self):
        """Stop acquisition if active, close serial ports."""
        if self._is_acquiring:
            self.stop_acquisition()

        self._stop_listener()

        if self._ser_in and self._ser_in != self._ser_out:
            self._ser_in.close()
        if self._ser_out:
            self._ser_out.close()

        self._ser_in = None
        self._ser_out = None
        self._is_connected = False
        logger.info("Disconnected")

    # ------------------------------------------------------------------
    # Device Info Commands
    # ------------------------------------------------------------------
    def _get_firmware_version(self):
        """GetGlobal 0x00 -> firmware and hardware version."""
        for attempt in range(3):
            try:
                resp = self._send_command(DeviceCommand.GetGlobal, bytes([0x00]))
                if resp and resp.is_response and len(resp.data) >= 6:
                    self.firmware_version = f"{resp.data[2]}.{resp.data[3]:02d}"
                    self.hardware_version = f"{resp.data[4]}.{resp.data[5]:02d}"
                    return
            except TimeoutError:
                time.sleep(0.05)
        raise ConnectionError("Failed to get firmware version after 3 attempts")

    def _get_device_id(self):
        """GetGlobal 0x01 -> 4-char device name."""
        for attempt in range(3):
            try:
                resp = self._send_command(DeviceCommand.GetGlobal, bytes([0x01]))
                if resp and resp.data:
                    name_bytes = resp.data[1:min(5, len(resp.data))]
                    self.device_name = name_bytes.decode("ascii", errors="replace").strip('\x00')
                    return
            except TimeoutError:
                time.sleep(0.01)
        raise ConnectionError("Failed to get device ID after 3 attempts")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def get_configuration(self) -> DeviceConfig:
        """
        Read the full device configuration: global settings + all 20 channels.

        Returns:
            DeviceConfig with all channel configurations populated.
        """
        logger.info("Reading device configuration...")

        # Get global DAQ parameters
        resp = self._send_command_retry(DeviceCommand.GetParam,
                                         bytes([ParamId.CommonDAQ]))
        if not resp or len(resp.data) < 19:
            raise RuntimeError(f"Invalid global config response: {resp}")

        # Skip param ID byte (first byte of data is the echo of 0x01)
        self.config = DeviceConfig.from_bytes(resp.data[1:])

        # Get each channel config (1-20)
        self.config.channels = []
        for ch_idx in range(1, 21):
            try:
                ch_resp = self._send_command_retry(
                    DeviceCommand.GetParam,
                    bytes([ParamId.ChannelConfig, ch_idx])
                )
                if ch_resp and len(ch_resp.data) > 1:
                    ch_cfg = ChannelConfig.from_bytes(ch_resp.data[1:])
                    self.config.channels.append(ch_cfg)
                    logger.debug(f"  {ch_cfg}")
            except Exception as e:
                logger.warning(f"  Ch {ch_idx}: failed ({e})")
                break

        logger.info(f"Configuration: {self.config}")
        return self.config

    def get_battery_info(self) -> BatteryInfo:
        """Query battery voltage."""
        if not self._is_acquiring:
            resp = self._send_command_retry(DeviceCommand.GetParam,
                                             bytes([ParamId.BatteryStatus]))
            if resp and len(resp.data) >= 7:
                raw_voltage = (resp.data[5] << 8) | resp.data[6]
                self.battery.voltage = raw_voltage * 0.00244
        return self.battery

    # ------------------------------------------------------------------
    # Device Lock / Unlock
    # ------------------------------------------------------------------
    def unlock_device(self) -> bool:
        """Unlock the device for configuration changes."""
        data = bytes([ParamId.UnlockDevice]) + UNLOCK_KEY
        try:
            self._send_command_retry(DeviceCommand.SetParam, data)
            self._is_locked = False
            logger.info("Device unlocked")
            return True
        except Exception as e:
            logger.error(f"Unlock failed: {e}")
            return False

    def lock_device(self) -> bool:
        """Lock the device."""
        try:
            self._send_command_retry(DeviceCommand.SetParam,
                                      bytes([ParamId.LockDevice]))
            self._is_locked = True
            logger.info("Device locked")
            return True
        except Exception as e:
            logger.error(f"Lock failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Data Acquisition
    # ------------------------------------------------------------------
    def start_acquisition(self):
        """
        Lock device and begin streaming data.

        The device sends ReceiveData (0xB0) packets at ~125 packets/sec
        (8ms per packet).
        """
        if self._is_acquiring:
            logger.warning("Already acquiring")
            return

        if self.config is None:
            self.get_configuration()

        # Lock device first
        if not self._is_locked:
            self.lock_device()

        # Reset counters
        self._first_packet_id = None
        self._last_packet_count = 0
        self._dropped_packets = 0
        self._total_packets = 0
        self._data_queue.clear()

        # Switch parser to checksum mode for streaming data
        self._parser.uses_checksum = True

        # Send start command
        self._send_command_retry(DeviceCommand.SetState,
                                  bytes([AcquisitionState.Start]))
        self._is_acquiring = True

        # Enable watchdog
        self._enable_watchdog()

        logger.info(f"Acquisition started at {self.config.sample_rate}Hz")

    def stop_acquisition(self):
        """Stop data streaming."""
        if not self._is_acquiring:
            return

        self._disable_watchdog()
        self._is_acquiring = False

        try:
            self._send_command_retry(DeviceCommand.SetState,
                                      bytes([AcquisitionState.Stop]))
        except Exception as e:
            logger.warning(f"Stop command failed: {e}")

        logger.info(f"Acquisition stopped. Dropped packets: {self._dropped_packets}")

    def read_data(self, timeout: float = 1.0) -> Optional[DataSample]:
        """
        Read the next parsed data sample from the queue.

        Args:
            timeout: Max seconds to wait for data.

        Returns:
            DataSample or None if timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                return self._data_queue.popleft()
            except IndexError:
                time.sleep(0.001)
        return None

    def read_all_data(self) -> List[DataSample]:
        """Read all currently queued data samples."""
        samples = []
        while self._data_queue:
            try:
                samples.append(self._data_queue.popleft())
            except IndexError:
                break
        return samples

    def on_data(self, callback: Callable[[DataSample], None]):
        """Register a callback for each received data sample."""
        self._data_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Low-Level Command Interface
    # ------------------------------------------------------------------
    def _send_command(self, command: DeviceCommand, data: bytes = b"",
                      timeout_ms: int = DEFAULT_TIMEOUT_MS) -> Packet:
        """
        Send a command packet and wait for the response.

        The SDK temporarily disables checksum for outgoing commands
        (SendDirectCommand sets usesChecksum=false before sending).
        """
        if not self._ser_out or not self._ser_out.is_open:
            raise ConnectionError("Serial port not open")

        # Build packet without checksum (matches SDK SendDirectCommand)
        pkt = build_packet(command, data, use_checksum=False)

        logger.debug(f"TX: {pkt.hex(' ')}")

        # Clear previous response
        self._response_event.clear()
        self._last_response = None

        # Send
        self._ser_out.write(pkt)
        self._ser_out.flush()

        # Wait for response (we need to read it ourselves if listener isn't running)
        if self._listener_thread is None or not self._listener_thread.is_alive():
            # Blocking read for response
            return self._read_response_blocking(timeout_ms / 1000.0)

        # Wait on the event from listener thread
        if not self._response_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"No response to {command.name} within {timeout_ms}ms")

        with self._response_lock:
            return self._last_response

    def _send_command_retry(self, command: DeviceCommand, data: bytes = b"",
                             max_retries: int = MAX_RETRIES,
                             timeout_ms: int = COMMAND_TIMEOUT_MS) -> Packet:
        """Send command with retries on failure/NACK."""
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = self._send_command(command, data, timeout_ms)
                if resp and resp.command == DeviceCommand.NegativeAck:
                    raise RuntimeError("Device NACK'd command")
                return resp
            except Exception as e:
                last_err = e
                time.sleep(0.05)
        raise last_err

    def _read_response_blocking(self, timeout: float) -> Packet:
        """Read a response directly (used before listener thread starts)."""
        deadline = time.monotonic() + timeout
        buf = bytearray()

        while time.monotonic() < deadline:
            if self._ser_in and self._ser_in.in_waiting:
                chunk = self._ser_in.read(self._ser_in.in_waiting)
                buf.extend(chunk)
            elif self._ser_out != self._ser_in and self._ser_out.in_waiting:
                chunk = self._ser_out.read(self._ser_out.in_waiting)
                buf.extend(chunk)
            else:
                time.sleep(0.005)
                continue

            # Try to parse
            pkt = self._try_parse_response(buf)
            if pkt:
                return pkt

        raise TimeoutError(f"No response within {timeout}s (got {len(buf)} bytes: {buf.hex(' ')})")

    def _try_parse_response(self, buf: bytearray) -> Optional[Packet]:
        """Try to parse a single response packet from a byte buffer."""
        # Find sync byte
        while buf and buf[0] != SYNC_BYTE:
            buf.pop(0)

        if len(buf) < 2:
            return None

        sync = buf[0]
        header = buf[1]
        cmd_nibble = header & 0xF0
        length_nibble = header & 0x07
        is_response = bool(header & 0x08)

        if length_nibble <= 5:
            total_needed = 2 + length_nibble  # sync + header + data
            if len(buf) < total_needed:
                return None
            data = bytes(buf[2:2 + length_nibble])
            del buf[:total_needed]
            try:
                cmd = DeviceCommand(cmd_nibble)
            except ValueError:
                return None
            return Packet(command=cmd, data=data, is_response=is_response)

        elif length_nibble == 6:
            if len(buf) < 3:
                return None
            data_len = buf[2]
            total_needed = 3 + data_len
            if len(buf) < total_needed:
                return None
            data = bytes(buf[3:3 + data_len])
            del buf[:total_needed]
            try:
                cmd = DeviceCommand(cmd_nibble)
            except ValueError:
                return None
            return Packet(command=cmd, data=data, is_response=is_response)

        return None

    # ------------------------------------------------------------------
    # Listener Thread
    # ------------------------------------------------------------------
    def _start_listener(self):
        """Start the background thread that reads incoming packets."""
        if self._listener_thread and self._listener_thread.is_alive():
            return

        self._stop_event.clear()
        self._listener_thread = threading.Thread(
            target=self._listener_loop,
            name="BioRadio-Listener",
            daemon=True
        )
        self._listener_thread.start()

    def _stop_listener(self):
        """Stop the listener thread."""
        self._stop_event.set()
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2.0)
        self._listener_thread = None

    def _listener_loop(self):
        """Background loop: read serial data and feed to parser."""
        logger.debug("Listener thread started")
        while not self._stop_event.is_set():
            try:
                # Read from input port
                if self._ser_in and self._ser_in.is_open:
                    waiting = self._ser_in.in_waiting
                    if waiting > 0:
                        chunk = self._ser_in.read(min(waiting, 65536))
                        if chunk:
                            self._parser.feed(chunk)

                # Also read from output port if different (responses come here)
                if (self._ser_out and self._ser_out != self._ser_in
                        and self._ser_out.is_open):
                    waiting = self._ser_out.in_waiting
                    if waiting > 0:
                        chunk = self._ser_out.read(min(waiting, 65536))
                        if chunk:
                            self._parser.feed(chunk)

            except serial.SerialException as e:
                logger.error(f"Serial error in listener: {e}")
                break
            except Exception as e:
                logger.error(f"Listener error: {e}")

            time.sleep(0.001)  # ~1ms poll interval

        logger.debug("Listener thread stopped")

    # ------------------------------------------------------------------
    # Packet Dispatch
    # ------------------------------------------------------------------
    def _on_packet_received(self, pkt: Packet):
        """Called by the parser for every valid incoming packet."""
        if pkt.is_response:
            self._on_response_received(pkt)
            return

        if pkt.command == DeviceCommand.ReceiveData:
            self._process_data_packet(pkt)

    def _on_response_received(self, pkt: Packet):
        """Handle a command response packet."""
        with self._response_lock:
            self._last_response = pkt
        self._response_event.set()

    # ------------------------------------------------------------------
    # Data Packet Parsing
    # ------------------------------------------------------------------
    def _process_data_packet(self, pkt: Packet):
        """
        Parse a ReceiveData (0xB0) packet into a DataSample.

        Packet layout (per the SDK):
          [2 bytes: status word] [2 bytes: packet ID]
          -- repeated 2x: --
            [MEMS: 12 bytes if enabled]
            [BioPotential: bit-packed]
            [External sensors: Aux(2 bytes) + PulseOx(5 bytes) each]
        """
        if not self.config or len(pkt.data) < 4:
            return

        # Reset watchdog
        self._reset_watchdog()

        data = pkt.data

        # Packet ID: first 2 bytes (big-endian)
        raw_packet_id = (data[0] << 8) | data[1]

        # Track packet IDs for dropped packet detection
        if self._first_packet_id is None:
            self._first_packet_id = raw_packet_id

        packet_count = (raw_packet_id - self._first_packet_id) & 0xFFFF

        # Status word: bytes 2-3
        status_bytes = data[2:4]
        event_marker = bool(status_bytes[0] & 0x80)
        status_code_val = (status_bytes[0] & 0x70) >> 4
        status_value = ((status_bytes[0] & 0x0F) << 8) | status_bytes[1]

        if status_code_val == StatusCode.BatteryVoltage:
            self.battery.voltage = status_value * 0.00244

        # Dropped packet detection
        if (packet_count != self._last_packet_count + 1
                and packet_count > 0
                and not (packet_count == 0 and self._last_packet_count == 0)):
            dropped = packet_count - self._last_packet_count - 1
            if dropped < 0:
                dropped = 0xFFFF - self._last_packet_count + packet_count
            self._dropped_packets += dropped
            logger.debug(f"Dropped {dropped} packets "
                         f"(#{self._last_packet_count} -> #{packet_count})")

        self._last_packet_count = packet_count
        self._total_packets += 1

        # Calculate data region sizes
        mems_size = 12 if self.config.mems_enabled else 0
        bp_channels = self.config.enabled_biopotential
        samples_per_packet = self.config.sample_rate // 250  # samples per sub-packet
        total_bits = sum(ch.bit_resolution for ch in bp_channels) * samples_per_packet
        bp_size = math.ceil(total_bits / 8)
        aux_channels = self.config.enabled_auxiliary
        pox_channels = self.config.enabled_pulseox
        ext_size = len(aux_channels) * 2 + len(pox_channels) * 5

        sample = DataSample(
            packet_id=packet_count,
            timestamp=time.time(),
            battery_voltage=self.battery.voltage,
            event_marker=event_marker,
        )

        # The SDK processes 2 sub-packets per data packet
        offset = 4  # skip packet ID (2) + status word (2)
        for sub in range(2):
            # MEMS (skip for now, parsed but empty in SDK)
            offset += mems_size

            # BioPotential channels
            if bp_channels and offset + bp_size <= len(data):
                byte_pos = offset
                bit_pos = 0
                for s in range(samples_per_packet):
                    for ch in bp_channels:
                        try:
                            val = extract_biopotential_value(
                                data, byte_pos, bit_pos, ch.bit_resolution
                            )
                            sample.biopotential.setdefault(ch.channel_index, []).append(val)
                        except (IndexError, ValueError) as e:
                            logger.debug(f"BP extract error ch{ch.channel_index}: {e}")

                        total_bits_consumed = bit_pos + ch.bit_resolution
                        byte_pos += total_bits_consumed // 8
                        bit_pos = total_bits_consumed % 8
            offset += bp_size

            # External sensors
            ext_offset = offset
            for ch in aux_channels:
                if ext_offset + 2 <= len(data):
                    val = (data[ext_offset] << 8) | data[ext_offset + 1]
                    sample.auxiliary[ch.channel_index] = val
                ext_offset += 2

            for ch in pox_channels:
                if ext_offset + 5 <= len(data):
                    flags = data[ext_offset]
                    ppg = (data[ext_offset + 1] << 8) | data[ext_offset + 2]
                    hr = (data[ext_offset + 3] << 1) | ((data[ext_offset + 4] & 0x80) >> 7)
                    spo2 = data[ext_offset + 4] & 0x7F
                    sample.pulseox[ch.channel_index] = {
                        "hr": hr, "spo2": spo2, "ppg": ppg, "flags": flags
                    }
                ext_offset += 5

            offset = ext_offset

        # Queue the sample
        self._data_queue.append(sample)

        # Notify callbacks
        for cb in self._data_callbacks:
            try:
                cb(sample)
            except Exception as e:
                logger.error(f"Data callback error: {e}")

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------
    def _enable_watchdog(self):
        self._watchdog_enabled = True
        self._reset_watchdog()

    def _disable_watchdog(self):
        self._watchdog_enabled = False
        if self._watchdog_timer:
            self._watchdog_timer.cancel()
            self._watchdog_timer = None

    def _reset_watchdog(self):
        if not self._watchdog_enabled:
            return
        if self._watchdog_timer:
            self._watchdog_timer.cancel()
        self._watchdog_timer = threading.Timer(
            WATCHDOG_TIMEOUT_S, self._watchdog_expired
        )
        self._watchdog_timer.daemon = True
        self._watchdog_timer.start()

    def _watchdog_expired(self):
        logger.error("Watchdog expired! No data received for 5 seconds.")
        self._disable_watchdog()
        self._is_acquiring = False
        try:
            self.disconnect()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Context Manager
    # ------------------------------------------------------------------
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()

    def __repr__(self):
        status = "ACQUIRING" if self._is_acquiring else "CONNECTED" if self._is_connected else "DISCONNECTED"
        return f"BioRadio({self.port_out_name}, {status})"


# ---------------------------------------------------------------------------
# Convenience: LSL bridge (optional, if pylsl is available)
# ---------------------------------------------------------------------------
def create_lsl_outlet(config: DeviceConfig):
    """
    Create an LSL outlet for BioRadio BioPotential data.
    Requires pylsl to be installed.

    Returns:
        pylsl.StreamOutlet or None if pylsl not available.
    """
    try:
        import pylsl
    except ImportError:
        logger.warning("pylsl not installed. LSL streaming not available.")
        return None

    bp_channels = config.enabled_biopotential
    if not bp_channels:
        logger.warning("No enabled BioPotential channels for LSL.")
        return None

    info = pylsl.StreamInfo(
        name=f"BioRadio_{config.name}",
        type="EMG",
        channel_count=len(bp_channels),
        nominal_srate=config.sample_rate,
        channel_format="float32",
        source_id=f"bioradio_{config.name}"
    )

    # Add channel metadata
    chns = info.desc().append_child("channels")
    for ch in bp_channels:
        c = chns.append_child("channel")
        c.append_child_value("label", ch.name or f"Ch{ch.channel_index}")
        c.append_child_value("unit", "microvolts")
        c.append_child_value("type", "EMG")

    return pylsl.StreamOutlet(info)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    """Command-line entry point for quick testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="BioRadio Python Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bioradio.py --scan                    # Scan for COM ports
  python bioradio.py --in COM9 --out COM10     # Connect and acquire
  python bioradio.py --in COM9 --out COM10 --lsl  # Stream to LSL
  python bioradio.py --in COM9 --info          # Print device info only
        """
    )
    parser.add_argument("--scan", action="store_true",
                        help="Scan for available COM ports")
    parser.add_argument("--in", dest="port_in", default=None,
                        help="Input COM port (e.g. COM9)")
    parser.add_argument("--out", dest="port_out", default=None,
                        help="Output COM port (e.g. COM10)")
    parser.add_argument("--info", action="store_true",
                        help="Print device info and config, then exit")
    parser.add_argument("--lsl", action="store_true",
                        help="Stream data to LSL")
    parser.add_argument("--duration", type=float, default=0,
                        help="Acquire for N seconds (0=until Ctrl+C)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    if args.scan:
        ports = scan_for_bioradio(verbose=True)
        if not ports:
            print("\nNo BioRadio candidates found.")
            print("Make sure the device is paired/plugged in.")
        return

    if not args.port_in:
        # Try auto-scan
        print("No port specified. Scanning...")
        ports = scan_for_bioradio(verbose=True)
        if len(ports) >= 2:
            args.port_in = ports[0]
            args.port_out = ports[1]
            print(f"\nAuto-selected: IN={args.port_in}, OUT={args.port_out}")
        elif len(ports) == 1:
            args.port_in = ports[0]
            args.port_out = ports[0]
            print(f"\nAuto-selected single port: {args.port_in}")
        else:
            print("\nCould not auto-detect ports. Use --in and --out flags.")
            return

    radio = BioRadio(
        port_in=args.port_in,
        port_out=args.port_out or args.port_in
    )

    try:
        radio.connect()
        config = radio.get_configuration()

        print(f"\n{'='*60}")
        print(f"  Device: {radio.device_name}")
        print(f"  Firmware: {radio.firmware_version}")
        print(f"  Hardware: {radio.hardware_version}")
        print(f"  Sample Rate: {config.sample_rate} Hz")
        print(f"  Termination: {'Single-Ended' if config.is_single_ended else 'Differential'}")
        print(f"  Battery: {radio.get_battery_info().voltage:.2f}V "
              f"({radio.battery.percentage:.0f}%)")
        print(f"\n  Channels:")
        for ch in config.channels:
            if ch.type_code != ChannelTypeCode.NotConnected:
                print(f"    {ch}")
        print(f"{'='*60}\n")

        if args.info:
            return

        # Set up LSL if requested
        lsl_outlet = None
        if args.lsl:
            lsl_outlet = create_lsl_outlet(config)
            if lsl_outlet:
                print("LSL outlet created. Streaming data to LSL network.")

        # Start acquisition
        radio.start_acquisition()
        print("Acquiring data... (Ctrl+C to stop)\n")

        start_time = time.time()
        sample_count = 0

        while True:
            sample = radio.read_data(timeout=0.1)
            if sample:
                sample_count += 1

                # Push to LSL if available
                if lsl_outlet and sample.biopotential:
                    bp_chs = config.enabled_biopotential
                    for s_idx in range(len(next(iter(sample.biopotential.values())))):
                        lsl_sample = []
                        for ch in bp_chs:
                            vals = sample.biopotential.get(ch.channel_index, [])
                            lsl_sample.append(float(vals[s_idx]) if s_idx < len(vals) else 0.0)
                        lsl_outlet.push_sample(lsl_sample)

                # Print periodic status
                if sample_count % 125 == 0:  # ~every second
                    elapsed = time.time() - start_time
                    bp_vals = ""
                    if sample.biopotential:
                        first_ch = next(iter(sample.biopotential.values()))
                        bp_vals = f" BP[0]={first_ch[0] if first_ch else '?'}"
                    print(f"  t={elapsed:.1f}s | packets={sample_count} | "
                          f"dropped={radio.dropped_packets} | "
                          f"bat={sample.battery_voltage:.2f}V{bp_vals}")

            # Check duration
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                break

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        radio.stop_acquisition()
        radio.disconnect()
        print(f"\nDone. Total packets: {radio._total_packets}, "
              f"Dropped: {radio.dropped_packets}")


if __name__ == "__main__":
    main()
