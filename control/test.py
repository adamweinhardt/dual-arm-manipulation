#!/usr/bin/env python3
"""
ZMQ Grasping Data Debug Script
This script helps diagnose ZMQ connection and grasping data issues
"""

import zmq
import time
import json
import sys
import threading
from datetime import datetime


class ZMQDebugger:
    def __init__(self):
        self.context = zmq.Context()
        self.ports_to_check = [5556, 5557, 5559, 5560]  # Common ports from your code
        self.results = {}

    def check_port_connectivity(self, port, timeout=2.0):
        """Test if we can connect to a ZMQ socket on given port"""
        try:
            socket = self.context.socket(zmq.SUB)
            socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))  # timeout in ms
            socket.setsockopt(zmq.CONFLATE, 1)
            socket.setsockopt_string(zmq.SUBSCRIBE, "")
            socket.connect(f"tcp://127.0.0.1:{port}")

            print(f"✓ Successfully connected to port {port}")
            return socket
        except Exception as e:
            print(f"✗ Failed to connect to port {port}: {e}")
            return None

    def listen_for_messages(self, port, duration=5.0):
        """Listen for messages on a specific port for given duration"""
        print(f"\n--- Listening on port {port} for {duration}s ---")

        socket = self.check_port_connectivity(port)
        if not socket:
            return

        start_time = time.time()
        message_count = 0

        try:
            while time.time() - start_time < duration:
                try:
                    message = socket.recv_json(flags=zmq.NOBLOCK)
                    message_count += 1
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                    print(f"[{timestamp}] Message #{message_count} received:")
                    print(
                        f"  Keys: {list(message.keys()) if isinstance(message, dict) else 'Not a dict'}"
                    )

                    # Special handling for grasping data (port 5560)
                    if port == 5560 and isinstance(message, dict):
                        if "grasping_points" in message:
                            grasping_data = message["grasping_points"]
                            print(
                                f"  Grasping data for boxes: {list(grasping_data.keys())}"
                            )

                            # Show structure of first box's data
                            if grasping_data:
                                first_box = list(grasping_data.keys())[0]
                                box_data = grasping_data[first_box]
                                print(
                                    f"  Box {first_box} data keys: {list(box_data.keys())}"
                                )

                                # Check for expected fields
                                expected_fields = [
                                    "point1",
                                    "point2",
                                    "approach_point1",
                                    "approach_point2",
                                    "normal1",
                                    "normal2",
                                ]
                                for field in expected_fields:
                                    if field in box_data:
                                        print(f"    ✓ {field}: {box_data[field]}")
                                    else:
                                        print(f"    ✗ Missing: {field}")
                        else:
                            print(f"  ✗ No 'grasping_points' key found")
                            print(f"  Available keys: {list(message.keys())}")

                    # Show sample of message content (truncated)
                    message_str = json.dumps(message, indent=2)
                    if len(message_str) > 500:
                        message_str = message_str[:500] + "\n... (truncated)"
                    print(f"  Content preview:\n{message_str}")
                    print("-" * 50)

                except zmq.Again:
                    time.sleep(0.1)  # No message available, wait a bit
                    continue

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            socket.close()

        print(f"Total messages received on port {port}: {message_count}")
        return message_count

    def create_test_publisher(self, port=5560):
        """Create a test publisher to send fake grasping data"""
        print(f"\n--- Creating test publisher on port {port} ---")

        try:
            pub_socket = self.context.socket(zmq.PUB)
            pub_socket.bind(f"tcp://*:{port}")
            time.sleep(1)  # Let socket settle

            # Create fake grasping data
            test_message = {
                "timestamp": time.time(),
                "grasping_points": {
                    "0": {  # Box ID 0
                        "point1": [0.1, 0.2, 0.3],
                        "point2": [0.4, 0.5, 0.6],
                        "approach_point1": [0.05, 0.15, 0.4],
                        "approach_point2": [0.45, 0.55, 0.7],
                        "normal1": [0.0, 0.0, -1.0],
                        "normal2": [0.0, 0.0, 1.0],
                    }
                },
            }

            print("Publishing test message every 2 seconds...")
            print("Press Ctrl+C to stop")

            count = 0
            while True:
                count += 1
                test_message["timestamp"] = time.time()
                test_message["sequence"] = count

                pub_socket.send_json(test_message)
                print(f"Sent test message #{count}")
                time.sleep(2)

        except KeyboardInterrupt:
            print("\nStopping test publisher")
        except Exception as e:
            print(f"Error creating test publisher: {e}")
        finally:
            pub_socket.close()

    def run_full_diagnostic(self):
        """Run complete diagnostic"""
        print("=" * 60)
        print("ZMQ GRASPING DATA DIAGNOSTIC")
        print("=" * 60)

        # Test connectivity to all known ports
        print("\n1. Testing port connectivity:")
        for port in self.ports_to_check:
            socket = self.check_port_connectivity(port, timeout=1.0)
            if socket:
                socket.close()

        # Listen for existing data on grasping port
        print(f"\n2. Listening for existing grasping data:")
        messages_received = self.listen_for_messages(5560, duration=3.0)

        if messages_received == 0:
            print(f"\n3. No messages found. Testing with fake publisher:")

            # Start test publisher in background thread
            pub_thread = threading.Thread(
                target=self.create_test_publisher, daemon=True
            )
            pub_thread.start()

            time.sleep(2)  # Let publisher start

            # Try to receive our own test messages
            print("Now listening for test messages:")
            test_messages = self.listen_for_messages(5560, duration=5.0)

            if test_messages > 0:
                print(
                    "✓ ZMQ communication works! Issue is likely missing grasping data publisher."
                )
            else:
                print(
                    "✗ Even test messages failed. Check ZMQ installation or firewall."
                )
        else:
            print("✓ Grasping data is being published!")

        print(f"\n4. Recommendations:")
        if messages_received == 0:
            print("- Start the grasping data publisher process")
            print("- Check if pose estimation system is running")
            print("- Verify the publisher is using port 5560")
            print("- Look for processes that compute grasp points")
        else:
            print("- Grasping data is available")
            print("- Check message format matches your code expectations")
            print("- Verify robot IDs and box IDs are correct")


def main():
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        debugger = ZMQDebugger()

        if command == "listen":
            port = int(sys.argv[2]) if len(sys.argv) > 2 else 5560
            duration = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
            debugger.listen_for_messages(port, duration)

        elif command == "publish":
            port = int(sys.argv[2]) if len(sys.argv) > 2 else 5560
            debugger.create_test_publisher(port)

        elif command == "check":
            port = int(sys.argv[2]) if len(sys.argv) > 2 else 5560
            debugger.check_port_connectivity(port)

        else:
            print(
                "Unknown command. Use: listen, publish, check, or no argument for full diagnostic"
            )

    else:
        # Run full diagnostic
        debugger = ZMQDebugger()
        debugger.run_full_diagnostic()


if __name__ == "__main__":
    print("Usage:")
    print("  python zmq_debug.py                    # Full diagnostic")
    print("  python zmq_debug.py listen [port] [duration]  # Listen on port")
    print("  python zmq_debug.py publish [port]     # Create test publisher")
    print("  python zmq_debug.py check [port]       # Test connectivity")
    print()

    main()
