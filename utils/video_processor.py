import pandas as pd
from moviepy.editor import VideoFileClip, concatenate_videoclips
from config import Config

class VideoProcessor:
    def __init__(self):
        self.pre_context = Config.PRE_CONTEXT      # seconds before highlight
        self.post_context = Config.POST_CONTEXT    # seconds after highlight
        self.merge_threshold = Config.MERGE_THRESHOLD  # seconds to merge clips

    def create_highlight_video(self, video_path, timestamps_file, output_path):
        """
        Create highlight video from timestamps.
        Safely handles timestamps that are near or beyond video duration.
        """
        # Load video and get its duration
        video = VideoFileClip(video_path)
        duration = video.duration
        print(f"🎬 Video duration: {duration:.2f} seconds")

        # Load timestamps from Excel
        df = pd.read_excel(timestamps_file)
        timestamps = sorted(df["timestamp"].tolist())

        if not timestamps:
            raise Exception("No timestamps found to create highlights")

        # Filter timestamps that are within video duration (allow a tiny margin)
        valid_timestamps = [ts for ts in timestamps if ts <= duration + 0.5]
        if len(valid_timestamps) != len(timestamps):
            print(f"⚠️ Filtered out {len(timestamps)-len(valid_timestamps)} timestamps beyond video duration")

        if not valid_timestamps:
            raise Exception("No valid timestamps inside video duration")

        # Create raw segments with context windows, clamped to [0, duration]
        segments = []
        for ts in valid_timestamps:
            start = max(0, ts - self.pre_context)
            end = min(duration, ts + self.post_context)
            # Only add if segment has positive length
            if start < end:
                segments.append([start, end])

        # Merge overlapping or nearby segments
        merged = []
        for seg in segments:
            if not merged:
                merged.append(seg)
                continue
            prev = merged[-1]
            # If current segment starts within merge_threshold of previous end, merge them
            if seg[0] <= prev[1] + self.merge_threshold:
                prev[1] = max(prev[1], seg[1])
            else:
                merged.append(seg)

        print(f"📊 Created {len(merged)} highlight segments from {len(valid_timestamps)} timestamps")

        # Extract clips
        clips = []
        for i, (start, end) in enumerate(merged):
            print(f"  Clip {i+1}: {start:.2f}s → {end:.2f}s (duration {end-start:.2f}s)")
            try:
                clip = video.subclip(start, end)
                # Add smooth fade transitions
                clip = clip.fadein(0.5).fadeout(0.5)
                clips.append(clip)
            except Exception as e:
                print(f"⚠️ Failed to extract clip {start}-{end}: {e}")
                continue

        if not clips:
            raise Exception("No clips could be extracted")

        # Concatenate all clips
        print(f"🎬 Concatenating {len(clips)} clips...")
        final = concatenate_videoclips(clips, method="compose")

        # Write output
        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None
        )

        # Clean up
        video.close()
        final.close()
        for clip in clips:
            clip.close()

        print(f"✅ Highlight video saved to {output_path}")
        return output_path