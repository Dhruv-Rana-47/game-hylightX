import os
import math
from moviepy.editor import VideoFileClip


class VideoSegmenter:
    """
    Splits a full input video into fixed-duration clips.

    Important design choice:
    - Clips are created WITHOUT audio because they are only used for
      downstream frame extraction.
    - This avoids MoviePy temp audio files (.m4a), reduces failures,
      and makes segmentation faster and cleaner.
    """

    def __init__(self, clip_duration=5):
        self.clip_duration = clip_duration

    def segment_video(self, video_path, clips_folder):
        """
        Segment the input video into fixed-duration clips.

        Args:
            video_path (str): Path to original uploaded video
            clips_folder (str): Folder where clip files will be stored

        Returns:
            list[dict]: Ordered clip metadata list
        """
        os.makedirs(clips_folder, exist_ok=True)

        clip_metadata = []
        video = None

        try:
            video = VideoFileClip(video_path)
            total_duration = float(video.duration)

            if total_duration <= 0:
                raise ValueError("Video duration is invalid or zero")

            total_clips = math.ceil(total_duration / self.clip_duration)

            print("=" * 60)
            print("✂️ Starting video segmentation")
            print(f"📹 Input video: {video_path}")
            print(f"⏱️ Total duration: {total_duration:.2f}s")
            print(f"🧩 Clip duration: {self.clip_duration}s")
            print(f"📦 Expected clips: {total_clips}")
            print("=" * 60)

            for clip_index in range(total_clips):
                start_time = clip_index * self.clip_duration
                end_time = min(start_time + self.clip_duration, total_duration)

                if start_time >= total_duration:
                    break

                clip_id = f"clip_{clip_index:03d}"
                clip_filename = f"{clip_id}.mp4"
                clip_path = os.path.join(clips_folder, clip_filename)

                print(f"🎬 Creating {clip_filename} | {start_time:.2f}s -> {end_time:.2f}s")

                subclip = None
                try:
                    subclip = video.subclip(start_time, end_time)

                    # IMPORTANT:
                    # audio=False avoids creation of temp .m4a files
                    # since these clips are only for frame extraction
                    subclip.write_videofile(
                        clip_path,
                        codec="libx264",
                        audio=False,
                        verbose=False,
                        logger=None
                    )

                    if not os.path.exists(clip_path):
                        raise Exception(f"Clip file was not created: {clip_path}")

                    clip_metadata.append({
                        "clip_id": clip_id,
                        "clip_index": clip_index,
                        "clip_filename": clip_filename,
                        "clip_path": clip_path,
                        "start_time": round(start_time, 3),
                        "end_time": round(end_time, 3),
                        "duration": round(end_time - start_time, 3)
                    })

                except Exception as e:
                    print(f"⚠️ Failed to create {clip_filename}: {e}")
                    continue
                finally:
                    if subclip is not None:
                        try:
                            subclip.close()
                        except Exception:
                            pass

            print("=" * 60)
            print(f"✅ Segmentation complete. Created {len(clip_metadata)} clips.")
            print("=" * 60)

            return clip_metadata

        except Exception as e:
            raise Exception(f"Video segmentation failed: {str(e)}")

        finally:
            if video is not None:
                try:
                    video.close()
                except Exception:
                    pass