#!/usr/bin/env python3
"""
Monitor current training process memory usage and progress.
"""
import psutil
import os
import time
import subprocess
from pathlib import Path

def get_training_process():
    """Find the TRM training process with highest memory usage."""
    max_memory = 0
    target_proc = None

    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
        try:
            if proc.info['name'] == 'python.exe' or proc.info['name'] == 'python':
                cmdline = proc.info['cmdline']
                if cmdline and 'train_multitask' in ' '.join(cmdline):
                    memory_mb = proc.info['memory_info'].rss / (1024 * 1024)
                    if memory_mb > max_memory:
                        max_memory = memory_mb
                        target_proc = proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return target_proc

def monitor_process():
    """Monitor the training process dynamically."""
    print("Starting dynamic process monitoring...")

    try:
        last_pid = None

        while True:
            proc = get_training_process()

            if not proc:
                if last_pid:
                    print(f"Training process PID {last_pid} no longer found!")
                    print("Checking if training restarted...")
                    time.sleep(5)
                    continue
                else:
                    print("No training process found! Waiting...")
                    time.sleep(10)
                    continue

            current_pid = proc.pid
            if current_pid != last_pid:
                print(f"\nNow monitoring new process PID: {current_pid}")
                if last_pid:
                    print(f"(Previous process PID {last_pid} stopped)")
                last_pid = current_pid

            memory_gb = proc.memory_info().rss / (1024 ** 3)
            cpu_percent = proc.cpu_percent(interval=1)

            status = "NORMAL" if memory_gb < 25 else "HIGH" if memory_gb < 30 else "CRITICAL"
            print(f"PID {current_pid} | RAM: {memory_gb:.1f}GB | CPU: {cpu_percent:.1f}% | Status: {status}")

            if memory_gb > 30:  # 30GB critical threshold
                print("CRITICAL: Memory usage > 30GB!")
                print("Recommendations:")
                print("   1. Stop training (Ctrl+C)")
                print("   2. Use config_memory_optimized.json")
                print("   3. Reduce batch_size and max_seq_len")
                break

            time.sleep(30)  # Check every 30 seconds

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"Error monitoring process: {e}")

if __name__ == "__main__":
    print("Training Monitor Started")
    print("Monitoring memory usage every 30 seconds...")
    print("Press Ctrl+C to stop\n")

    monitor_process()