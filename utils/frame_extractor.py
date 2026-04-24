import os
import cv2
from config import Config


class FrameExtractor:
    """
    Clip-based frame extractor.

    This version extracts frames from each 5-second clip instead of extracting
    from the full video at low FPS.

    Key behavior:
    - Uses configurable high FPS from Config.HIGH_FPS
    - Can optionally fall back to native clip FPS
    - Preserves exact clip-relative and absolute video timestamps
    - Saves frames in per-clip folders for better organization
    """

    def __init__(self):
        self.target_fps = Config.HIGH_FPS

    def _safe_video_fps(self, cap):
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            return 30.0
        return float(fps)

    def extract_frames_from_clip(self, clip_info, base_frames_folder):
        """
        Extract high-FPS frames from one clip.

        Args:
            clip_info (dict): clip metadata from VideoSegmenter
            base_frames_folder (str): base folder for this job's frames

        Returns:
            list[dict]: ordered frame metadata
        """
        clip_id = clip_info["clip_id"]
        clip_index = clip_info["clip_index"]
        clip_path = clip_info["clip_path"]
        clip_start_time = float(clip_info["start_time"])

        clip_frames_folder = os.path.join(base_frames_folder, clip_id)
        os.makedirs(clip_frames_folder, exist_ok=True)

        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            raise Exception(f"Could not open clip: {clip_path}")

        native_fps = self._safe_video_fps(cap)

        # If HIGH_FPS is set, use min(HIGH_FPS, native_fps) to avoid fake oversampling.
        # If HIGH_FPS is 0 or None, use native FPS.
        if self.target_fps and self.target_fps > 0:
            extraction_fps = min(float(self.target_fps), native_fps)
        else:
            extraction_fps = native_fps

        if extraction_fps <= 0:
            extraction_fps = native_fps

        frame_interval = max(1, int(round(native_fps / extraction_fps)))

        print("=" * 60)
        print(f"🖼️ Extracting frames from {clip_id}")
        print(f"📹 Clip path: {clip_path}")
        print(f"🎞️ Native FPS: {native_fps:.2f}")
        print(f"⚙️ Extraction FPS: {extraction_fps:.2f}")
        print(f"📏 Frame interval: every {frame_interval} frame(s)")
        print("=" * 60)

        frames_info = []
        raw_frame_index = 0
        saved_frame_index = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if raw_frame_index % frame_interval == 0:
                    clip_timestamp = raw_frame_index / native_fps
                    video_timestamp = clip_start_time + clip_timestamp

                    filename = (
                        f"{clip_id}_frame_{saved_frame_index:05d}_"
                        f"{video_timestamp:.2f}.jpg"
                    )
                    frame_path = os.path.join(clip_frames_folder, filename)

                    height, width = frame.shape[:2]
                    if height > 720:
                        scale = 720 / height
                        new_width = int(width * scale)
                        frame = cv2.resize(frame, (new_width, 720))

                    success = cv2.imwrite(frame_path, frame)
                    if not success:
                        print(f"⚠️ Failed to save frame: {frame_path}")
                        raw_frame_index += 1
                        continue

                    frames_info.append({
                        "frame": filename,
                        "filename": filename,  # keep compatibility with old code
                        "path": frame_path,
                        "clip_id": clip_id,
                        "clip_index": clip_index,
                        "frame_index": saved_frame_index,
                        "raw_frame_index": raw_frame_index,
                        "clip_timestamp": round(clip_timestamp, 3),
                        "timestamp": round(video_timestamp, 3),  # compatibility
                        "video_timestamp": round(video_timestamp, 3)
                    })

                    saved_frame_index += 1

                raw_frame_index += 1

        finally:
            cap.release()

        print(f"✅ Extracted {len(frames_info)} frames from {clip_id}")
        return frames_info

    def extract_frames_from_clips(self, clips_info, base_frames_folder):
        """
        Extract frames from all clips in order.

        Args:
            clips_info (list[dict]): list of clip metadata
            base_frames_folder (str): root frames folder for current job

        Returns:
            list[dict]: all frame metadata sorted by original video timestamp
        """
        all_frames = []

        if not clips_info:
            raise Exception("No clips available for frame extraction")

        for clip_info in sorted(clips_info, key=lambda x: x["clip_index"]):
            clip_frames = self.extract_frames_from_clip(clip_info, base_frames_folder)
            all_frames.extend(clip_frames)

        # Deterministic sort by video timestamp, then clip index, then frame index
        all_frames.sort(
            key=lambda x: (
                x["video_timestamp"],
                x["clip_index"],
                x["frame_index"]
            )
        )

        print("=" * 60)
        print(f"✅ Total extracted frames from all clips: {len(all_frames)}")
        print("=" * 60)

        return all_frames