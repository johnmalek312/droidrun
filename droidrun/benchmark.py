import time
import asyncio
import os
import json
import pyperclip
import statistics  # Import statistics for calculating the average
from .tools import Tools, DeviceManager

# --- Measure time for imports ---
start_time_imports = time.time()
# Imports are already present, no need to duplicate
end_time_imports = time.time()
print(f"Import time: {end_time_imports - start_time_imports:.4f} seconds")
# --- End Measure time for imports ---

# --- Measure time for DeviceManager instantiation ---
start_time_dev_manager = time.time()
device_manager = DeviceManager()
end_time_dev_manager = time.time()
print(f"DeviceManager instantiation time: {end_time_dev_manager - start_time_dev_manager:.4f} seconds")
# --- End Measure time for DeviceManager instantiation ---

async def main():
    # --- Measure time for list_devices ---
    start_time_list_devices = time.time()
    devices = await device_manager.list_devices()
    end_time_list_devices = time.time()
    print(f"list_devices time: {end_time_list_devices - start_time_list_devices:.4f} seconds")
    # --- End Measure time for list_devices ---

    if not devices:
        print("No devices connected.")
        return

    device_serial = devices[0].serial
    print(f"Using device: {device_serial}")

    tools = Tools(serial=device_serial)

    # --- Benchmark get_clickables ---
    iterations_to_run = 10
    successful_runs = 0
    execution_times = []
    total_attempts = 0

    print(f"\nStarting benchmark for get_clickables ({iterations_to_run} successful runs)...")

    while successful_runs < iterations_to_run:
        total_attempts += 1
        print(f"\nAttempt {total_attempts} (Successful: {successful_runs}/{iterations_to_run})")
        try:
            start_time_clickable = time.perf_counter() # Use perf_counter for more precise timing
            data = await tools.get_clickables()
            end_time_clickable = time.perf_counter()

            duration = end_time_clickable - start_time_clickable
            execution_times.append(duration)
            successful_runs += 1
            print(f"  Success! Time taken: {duration:.4f} seconds")
            # Optional: print data length or some info
            # print(f"  Clickables found: {len(data)}")

        except Exception as e:
            print(f"  Attempt failed with error: {e}")
            # Optional: add a small delay before retrying
            # await asyncio.sleep(0.5)

    print(f"\nBenchmark finished after {total_attempts} total attempts.")
    print(f"--- Results for {iterations_to_run} successful get_clickables runs ---")
    for i, t in enumerate(execution_times):
        print(f"  Run {i+1}: {t:.4f} seconds")

    if execution_times:
        average_time = statistics.mean(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        print(f"\nAverage time: {average_time:.4f} seconds")
        print(f"Min time: {min_time:.4f} seconds")
        print(f"Max time: {max_time:.4f} seconds")
    # --- End Benchmark get_clickables ---


if __name__ == "__main__":
    asyncio.run(main())