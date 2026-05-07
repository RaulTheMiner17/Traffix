import subprocess
import sys
import os

# Define our directories based on where this script is located
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMO_TEST_DIR = os.path.join(ROOT_DIR, "sumo_test")

print("\n" + "="*50)
print("🚦 STARTING SIDE-BY-SIDE TRAFFIC SIMULATION COMPARISON")
print("="*50 + "\n")

try:
    # 1. Launch the AI Model
    # We set cwd=SUMO_TEST_DIR so it knows to look inside the sumo_test folder for its files
    print("[INFO] Launching AI Model (demo_model.py)...")
    ai_process = subprocess.Popen(
        [sys.executable, "demo_model.py"],
        cwd=SUMO_TEST_DIR, 
        creationflags=subprocess.CREATE_NEW_CONSOLE # Opens a new CMD window
    )

    # 2. Launch the Static Baseline Model
    # We set cwd=ROOT_DIR because static_sumo_controller.py handles its own paths
    print("[INFO] Launching Static Model (static_sumo_controller.py)...")
    static_process = subprocess.Popen(
        [sys.executable, "static_sumo_controller.py"],
        cwd=ROOT_DIR,
        creationflags=subprocess.CREATE_NEW_CONSOLE # Opens a second CMD window
    )

    print("\n✅ Both simulations have been launched in separate windows!")
    print("👉 Arrange the two black console windows side-by-side on your screen.")
    print("👉 Watch them step through the simulation simultaneously.")
    print("\nPress Ctrl+C here to terminate both simulations early if needed.")

    # Keep the main script running until both windows are closed
    ai_process.wait()
    static_process.wait()

except KeyboardInterrupt:
    print("\n[INFO] Terminating both simulations...")
    ai_process.terminate()
    static_process.terminate()
    print("Done.")