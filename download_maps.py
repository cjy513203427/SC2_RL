"""
Download StarCraft II Mini-game Maps
This script downloads required mini-game maps for PySC2
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path

try:
    import requests
except ImportError:
    print("Installing requests library...")
    os.system("pip install requests")
    import requests


def download_file(url, destination):
    """Download a file from URL to destination"""
    print(f"Downloading from: {url}")
    print(f"Destination: {destination}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    downloaded = 0
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    
    print("\n[OK] Download completed!")
    return True


def extract_zip(zip_path, extract_to):
    """Extract a zip file"""
    print(f"Extracting to: {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("[OK] Extraction completed!")


def download_mini_games():
    """Download and install SC2 mini-game maps"""
    
    # Get SC2 path
    sc2_path = os.environ.get('SC2PATH')
    if not sc2_path:
        default_path = r"C:\Program Files (x86)\StarCraft II"
        if os.path.exists(default_path):
            sc2_path = default_path
        else:
            print("[ERROR] SC2PATH not set and default path not found!")
            print("Please run setup_sc2_env.py first")
            return False
    
    print(f"SC2 installation path: {sc2_path}")
    
    # Map pack URLs - from official sources
    # Reference: https://github.com/Blizzard/s2client-proto#downloads
    map_packs = {
        "mini_games": "https://github.com/deepmind/pysc2/releases/download/v1.2/mini_games.zip"
    }
    
    maps_dir = Path(sc2_path) / "Maps"
    maps_dir.mkdir(exist_ok=True)
    
    temp_dir = Path("temp_maps")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        for pack_name, url in map_packs.items():
            print(f"\n{'=' * 60}")
            print(f"Downloading {pack_name} map pack...")
            print('=' * 60)
            
            # Download
            zip_file = temp_dir / f"{pack_name}.zip"
            try:
                download_file(url, zip_file)
            except Exception as e:
                print(f"[ERROR] Failed to download: {e}")
                continue
            
            # Extract
            extract_zip(zip_file, maps_dir)
            
            # Clean up
            zip_file.unlink()
            
            print(f"[OK] {pack_name} maps installed successfully!")
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        # List installed maps
        print(f"\n{'=' * 60}")
        print("Installed maps:")
        print('=' * 60)
        
        mini_games_dir = maps_dir / "mini_games"
        if mini_games_dir.exists():
            maps = list(mini_games_dir.glob("*.SC2Map"))
            for map_file in maps:
                print(f"  - {map_file.name}")
            print(f"\nTotal: {len(maps)} mini-game maps")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("SC2 Mini-game Maps Downloader")
    print("=" * 60)
    
    success = download_mini_games()
    
    if success:
        print("\n[SUCCESS] All maps downloaded successfully!")
        print("\nYou can now run:")
        print("  python test_sc2_env.py")
    else:
        print("\n[FAILED] Map download failed!")
        sys.exit(1)

