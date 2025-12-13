import os
import shutil
import random
from datetime import datetime, timedelta

# =================CONFIGURATION=================
# 1. source_path: The real, good data path
SOURCE_PATH = "/mnt/c/kitti_data/2011_09_30/2011_09_30_drive_0033_sync"

# 2. output_base: Where to save the synthetic data
OUTPUT_BASE = "./bad_data_test" 

# 3. Corruption Mode: 'PARTIAL_FAILURE'
# 'PARTIAL_FAILURE': Randomly corrupts 40% of frames to achieve ~60% Success Rate
# 'OFFSET': (Previous) Constant delay for 0% Success Rate
# 'JITTER': (Previous) Random noise
MODE = 'PARTIAL_FAILURE' 
# ===============================================

def parse_kitti_time(line):
    # KITTI format example: 2011-09-26 13:02:25.964809000
    # Python standard lib only supports 6 decimal places (microseconds)
    clean_line = line.strip()
    timestamp_str = clean_line[:-3] 
    dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
    return dt

def format_kitti_time(dt):
    # Convert back to string and add fake nanoseconds (000)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f") + "000"

def corrupt_timestamps(src_file, dst_file, mode):
    print(f"Processing: {src_file} -> {dst_file}")
    
    with open(src_file, 'r') as f_in, open(dst_file, 'w') as f_out:
        lines = f_in.readlines()
        
        for line in lines:
            if not line.strip(): continue
            
            dt = parse_kitti_time(line)
            
            # --- APPLY CORRUPTION LOGIC ---
            if mode == 'PARTIAL_FAILURE':
                # Generate a random float between 0.0 and 1.0
                # If number is less than 0.40 (40%), we corrupt the data.
                # This leaves ~60% of data clean (Success Rate ~60%).
                if random.random() < 0.40:
                    # Add 50ms delay (Above the 30ms threshold -> FAIL)
                    dt += timedelta(milliseconds=50)
                else:
                    # Keep original data (PASS)
                    pass

            elif mode == 'OFFSET':
                # Constant delay -> 0% Success
                dt += timedelta(milliseconds=45) 
                
            elif mode == 'JITTER':
                # Random noise -> Unpredictable Success
                noise_ms = random.uniform(-30, 30)
                dt += timedelta(milliseconds=noise_ms)
            
            # Write the timestamp back to file
            f_out.write(format_kitti_time(dt) + "\n")

def main():
    # 1. Define folder names
    folder_name = f"drive_0033_CORRUPTED_{MODE}"
    destination_path = os.path.join(OUTPUT_BASE, folder_name)
    
    print(f"--- Generating Synthetic Data (Target: ~60% Success Rate) ---")
    
    # 2. Create the folder structure
    os.makedirs(os.path.join(destination_path, "image_02"), exist_ok=True)
    os.makedirs(os.path.join(destination_path, "image_03"), exist_ok=True)
    
    # 3. Copy image_02 (Control Group - Clean)
    src_02 = os.path.join(SOURCE_PATH, "image_02", "timestamps.txt")
    dst_02 = os.path.join(destination_path, "image_02", "timestamps.txt")
    shutil.copy(src_02, dst_02)
    print(f"Copied original (clean): {src_02}")
    
    # 4. Generate image_03 (Test Group - Mixed Clean/Corrupted)
    src_03 = os.path.join(SOURCE_PATH, "image_03", "timestamps.txt")
    dst_03 = os.path.join(destination_path, "image_03", "timestamps.txt")
    corrupt_timestamps(src_03, dst_03, MODE)
    print(f"Generated mixed corrupted file: {dst_03}")
    
    print("-" * 50)
    print("DONE! Run this command to verify the 60% success rate:")
    print(f"\npython3 sync_validator.py --path \"{os.path.abspath(destination_path)}\"\n")

if __name__ == "__main__":
    main()
