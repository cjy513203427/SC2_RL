"""
StarCraft II Environment Setup Script
This script configures the SC2 environment path for PySC2
"""

import os
import sys

def setup_sc2_path():
    """Configure SC2 installation path for Windows"""
    
    # Default SC2 installation path on Windows
    default_paths = [
        r"C:\Program Files (x86)\StarCraft II",
        r"C:\Program Files\StarCraft II",
        r"D:\StarCraft II",
        r"E:\StarCraft II"
    ]
    
    # Check which path exists
    sc2_path = None
    for path in default_paths:
        if os.path.exists(path):
            sc2_path = path
            print(f"Found StarCraft II at: {sc2_path}")
            break
    
    if not sc2_path:
        print("StarCraft II installation not found in default locations.")
        print("Please specify the path manually:")
        custom_path = input("Enter SC2 installation path: ").strip()
        if os.path.exists(custom_path):
            sc2_path = custom_path
        else:
            print(f"Path {custom_path} does not exist!")
            return False
    
    # Set environment variable
    os.environ['SC2PATH'] = sc2_path
    print(f"Environment variable SC2PATH set to: {sc2_path}")
    
    # Verify the installation
    versions_dir = os.path.join(sc2_path, "Versions")
    if os.path.exists(versions_dir):
        versions = [v for v in os.listdir(versions_dir) if os.path.isdir(os.path.join(versions_dir, v))]
        print(f"Found SC2 versions: {versions}")
        return True
    else:
        print("Warning: Versions directory not found!")
        return False

if __name__ == "__main__":
    success = setup_sc2_path()
    if success:
        print("\n[SUCCESS] SC2 environment setup completed successfully!")
        print("\nTo use PySC2, you can now run:")
        print("  python test_sc2_env.py")
    else:
        print("\n[FAILED] SC2 environment setup failed!")
        sys.exit(1)

