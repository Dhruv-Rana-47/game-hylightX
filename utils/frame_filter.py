import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from config import Config


class FrameFilter:
    """
    Smart frame filter for removing redundant frames while preserving transitions.

    Added behavior:
    - After filtering a clip, all unselected frame files can be deleted
    - Remaining files in frames/<job_id>/<clip_id>/ are exactly the selected frames
    """

    def __init__(self):
        self.frame_diff_threshold = Config.FRAME_DIFF_THRESHOLD
        self.ssim_diff_threshold = Config.SSIM_DIFF_THRESHOLD
        self.keep_first_frame = Config.KEEP_FIRST_FRAME
        self.keep_last_frame = Config.KEEP_LAST_FRAME
        self.pre_change_frames = Config.PRE_CHANGE_FRAMES
        self.min_selected_per_clip = Config.MIN_SELECTED_PER_CLIP
        self.max_selected_per_clip = Config.MAX_SELECTED_PER_CLIP

    def _load_frame_gray(self, frame_path):
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Frame file not found: {frame_path}")

        image = cv2.imread(frame_path)
        if image is None:
            raise ValueError(f"Could not read frame: {frame_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))
        return gray

    def compute_frame_difference(self, frame_a_path, frame_b_path):
        gray_a = self._load_frame_gray(frame_a_path)
        gray_b = self._load_frame_gray(frame_b_path)

        abs_diff = cv2.absdiff(gray_a, gray_b)
        pixel_diff = float(np.mean(abs_diff) / 255.0)

        ssim_score = float(ssim(gray_a, gray_b))
        ssim_diff = 1.0 - ssim_score

        combined_diff = max(pixel_diff, ssim_diff)

        return {
            "pixel_diff": round(pixel_diff, 4),
            "ssim_score": round(ssim_score, 4),
            "ssim_diff": round(ssim_diff, 4),
            "combined_diff": round(combined_diff, 4)
        }

    def _is_significant_change(self, diff_metrics):
        return (
            diff_metrics["pixel_diff"] >= self.frame_diff_threshold
            or diff_metrics["ssim_diff"] >= self.ssim_diff_threshold
        )

    def filter_clip_frames(self, clip_frames):
        if not clip_frames:
            return []

        clip_frames = sorted(
            clip_frames,
            key=lambda x: (x["clip_timestamp"], x["frame_index"])
        )

        if len(clip_frames) == 1:
            return clip_frames[:]

        selected_indices = set()

        if self.keep_first_frame:
            selected_indices.add(0)

        for i in range(1, len(clip_frames)):
            prev_frame = clip_frames[i - 1]
            curr_frame = clip_frames[i]

            try:
                diff_metrics = self.compute_frame_difference(
                    prev_frame["path"],
                    curr_frame["path"]
                )
            except Exception as e:
                print(f"⚠️ Frame comparison failed for {prev_frame['frame']} -> {curr_frame['frame']}: {e}")
                selected_indices.add(i)
                continue

            if self._is_significant_change(diff_metrics):
                for back in range(self.pre_change_frames, 0, -1):
                    pre_idx = i - back
                    if pre_idx >= 0:
                        selected_indices.add(pre_idx)

                selected_indices.add(i)

        if self.keep_last_frame:
            selected_indices.add(len(clip_frames) - 1)

        selected = [clip_frames[i] for i in sorted(selected_indices)]

        if len(selected) < self.min_selected_per_clip:
            fallback_indices = {0, len(clip_frames) - 1}
            selected = [clip_frames[i] for i in sorted(fallback_indices)]

        if len(selected) > self.max_selected_per_clip:
            selected = self._downsample_selected_frames(selected)

        return selected

    def _downsample_selected_frames(self, selected_frames):
        if len(selected_frames) <= self.max_selected_per_clip:
            return selected_frames

        keep_positions = np.linspace(
            0,
            len(selected_frames) - 1,
            self.max_selected_per_clip,
            dtype=int
        )
        return [selected_frames[i] for i in keep_positions]

    def delete_unselected_frames(self, clip_frames, selected_frames):
        """
        Deletes frame image files that were not selected.
        """
        selected_paths = {frame["path"] for frame in selected_frames}
        deleted_count = 0

        for frame in clip_frames:
            frame_path = frame["path"]
            if frame_path not in selected_paths and os.path.exists(frame_path):
                try:
                    os.remove(frame_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to delete {frame_path}: {e}")

        return deleted_count

    def filter_and_cleanup_clip(self, clip_frames):
        """
        Filter one clip and delete unselected frame files.
        """
        selected_frames = self.filter_clip_frames(clip_frames)
        deleted_count = self.delete_unselected_frames(clip_frames, selected_frames)
        return selected_frames, deleted_count