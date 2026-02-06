"""
LSL Diagnostic Script
=====================
Tests LSL stream creation and discovery.

Run this to check if LSL is working properly.
"""

import time
import threading

try:
    import pylsl
    print(f"pylsl version: {pylsl.__version__}")
except ImportError:
    print("ERROR: pylsl not installed!")
    print("Install with: pip install pylsl")
    exit(1)

def create_test_stream():
    """Create a simple test stream."""
    print("\n=== Creating Test Stream ===")
    
    info = pylsl.StreamInfo(
        name="TestStream",
        type="EMG",
        channel_count=8,
        nominal_srate=200,
        channel_format=pylsl.cf_float32,
        source_id="test123"
    )
    
    outlet = pylsl.StreamOutlet(info)
    print(f"Created stream: {info.name()}")
    print(f"  Type: {info.type()}")
    print(f"  Channels: {info.channel_count()}")
    print(f"  Rate: {info.nominal_srate()} Hz")
    
    return outlet

def discover_streams():
    """Discover all available LSL streams."""
    print("\n=== Discovering Streams ===")
    print("Scanning for 3 seconds...")
    
    # Note: parameter name varies by pylsl version
    try:
        # Try positional first (older API)
        streams = pylsl.resolve_streams(3.0)
    except TypeError:
        # Try keyword (newer API)
        streams = pylsl.resolve_streams(wait_time=3.0)
    
    if streams:
        print(f"\nFound {len(streams)} stream(s):")
        for i, stream in enumerate(streams):
            print(f"\n  [{i+1}] {stream.name()}")
            print(f"      Type: {stream.type()}")
            print(f"      Channels: {stream.channel_count()}")
            print(f"      Rate: {stream.nominal_srate()} Hz")
            print(f"      Source ID: {stream.source_id()}")
            print(f"      Hostname: {stream.hostname()}")
    else:
        print("\nNo streams found!")
        print("\nPossible causes:")
        print("  1. No LSL streams are running")
        print("  2. Firewall blocking LSL (UDP ports 16571-16600)")
        print("  3. Network adapter issues with multicast")
    
    return streams

def push_data(outlet, duration=5):
    """Push data to the stream."""
    print(f"\nPushing data for {duration} seconds...")
    import numpy as np
    
    count = 0
    start = time.time()
    while time.time() - start < duration:
        sample = np.random.randn(8).tolist()
        outlet.push_sample(sample)
        count += 1
        time.sleep(1/200)  # 200 Hz
    
    print(f"Pushed {count} samples")

def receive_data(stream_name, duration=3):
    """Try to receive data from a stream."""
    print(f"\n=== Receiving from '{stream_name}' ===")
    
    try:
        # Try positional first (older API)
        streams = pylsl.resolve_byprop("name", stream_name, 1, 2.0)
    except TypeError:
        # Try keyword (newer API)
        streams = pylsl.resolve_byprop("name", stream_name, timeout=2.0)
    
    if not streams:
        print(f"Stream '{stream_name}' not found!")
        return
    
    inlet = pylsl.StreamInlet(streams[0])
    print(f"Connected to {stream_name}")
    
    count = 0
    start = time.time()
    while time.time() - start < duration:
        samples, timestamps = inlet.pull_chunk(timeout=0.1)
        if samples:
            count += len(samples)
    
    print(f"Received {count} samples in {duration} seconds")

def main():
    print("=" * 50)
    print("LSL Diagnostic Tool")
    print("=" * 50)
    
    # First, just scan for any existing streams
    print("\n[Step 1] Checking for existing streams...")
    existing = discover_streams()
    
    # Create our own test stream
    print("\n[Step 2] Creating test stream...")
    outlet = create_test_stream()
    
    # Give it a moment to register
    time.sleep(0.5)
    
    # Start pushing data in background
    push_thread = threading.Thread(target=push_data, args=(outlet, 5), daemon=True)
    push_thread.start()
    
    # Wait a bit then scan again
    time.sleep(1)
    
    print("\n[Step 3] Re-scanning for streams (should find TestStream)...")
    streams = discover_streams()
    
    # Try to receive from our own stream
    if any(s.name() == "TestStream" for s in streams):
        receive_data("TestStream", duration=2)
        print("\n✓ LSL is working correctly!")
    else:
        print("\n✗ Could not find our own test stream!")
        print("  This suggests a network/firewall issue.")
        print("\nTroubleshooting:")
        print("  1. Check Windows Firewall - allow Python through")
        print("  2. Try disabling firewall temporarily")
        print("  3. Check if antivirus is blocking UDP multicast")
    
    push_thread.join()
    print("\nDone!")

if __name__ == "__main__":
    main()
