import numpy as np
import sys

def inspect_npz(filename):
    try:
        # Load the file
        data = np.load(filename, allow_pickle=True)
        
        print(f"--- Inspecting: {filename} ---")
        print(f"Keys found: {list(data.keys())}\n")

        # Loop through each key and print details
        for key in data.keys():
            item = data[key]
            print(f"Key: '{key}'")
            
            # Check if it's a scalar (0-d array) or standard array
            if item.ndim == 0:
                print(f"  Type: Scalar")
                print(f"  Value: {item}")
            else:
                print(f"  Shape: {item.shape}")
                print(f"  Data type: {item.dtype}")
                # Print the first few items if it's large, or all if small
                if item.size > 10:
                    print(f"  Data (first 5): {item.flatten()[:5]} ...")
                else:
                    print(f"  Data: {item}")
            print("-" * 30)
            
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    # Change this to your actual filename
    filename = "experiments/PID_ff/logs/complex/seq_figure8_complex_PID_ff_bw_20251210-150257_L.npz"
    inspect_npz(filename)