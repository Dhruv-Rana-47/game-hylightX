import os
import shutil
from config import Config


class CleanupManager:
    def __init__(self):
        self.folders = [
            Config.UPLOAD_FOLDER,
            Config.FRAMES_FOLDER,
            Config.OUTPUT_FOLDER,
            Config.CLIPS_FOLDER
        ]

    def clear_all(self):
        """
        Deletes all previous pipeline data before a new processing run.
        Prevents reuse of stale frames, clips, output JSONs, and videos.
        """
        print("🧹 Cleaning old data...")

        for folder in self.folders:
            if os.path.exists(folder):
                try:
                    shutil.rmtree(folder)
                    print(f"🗑️ Deleted: {folder}")
                except Exception as e:
                    print(f"⚠️ Failed to delete {folder}: {e}")

            os.makedirs(folder, exist_ok=True)

        print("✅ Cleanup complete. Fresh environment ready.")