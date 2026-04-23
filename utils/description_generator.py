import os
import json
import time
from google import genai
from google.genai import types
from config import Config


class DescriptionGenerator:
    def __init__(self):
        self.api_key = Config.API_KEY
        self.fallback_keys = Config.FALLBACK_API_KEYS
        self.models = Config.MODELS
        self.request_delay = Config.REQUEST_DELAY
        self.current_model_index = 0
        self.current_key_index = 0
        self.client = None
        self.init_client()

    def init_client(self):
        current_key = self.api_key if self.current_key_index == 0 else self.fallback_keys[self.current_key_index - 1]
        self.client = genai.Client(api_key=current_key)

    def switch_api_key(self):
        if self.current_key_index < len(self.fallback_keys):
            self.current_key_index += 1
            self.init_client()
            print(f"🔄 Switched to API key {self.current_key_index}")
            return True
        return False

    def is_rate_limit_error(self, e):
        msg = str(e).lower()
        return "quota" in msg or "rate" in msg or "limit" in msg or "429" in msg

    def _safe_read_json(self, file_path, default_value):
        if not os.path.exists(file_path):
            return default_value
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default_value

    def _write_json(self, file_path, data):
        temp_path = f"{file_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, file_path)

    def _load_memory_context(self, memory_file):
        memory_data = self._safe_read_json(memory_file, {
            "clips": [],
            "video_summary": ""
        })
        return memory_data.get("video_summary", "").strip(), memory_data

    def _append_descriptions(self, description_file, new_items):
        existing = self._safe_read_json(description_file, [])

        existing_keys = {
            (item.get("clip_id"), item.get("frame"), item.get("timestamp"))
            for item in existing
        }

        for item in new_items:
            key = (item.get("clip_id"), item.get("frame"), item.get("timestamp"))
            if key not in existing_keys:
                existing.append(item)
                existing_keys.add(key)

        existing.sort(key=lambda x: (x.get("timestamp", 0), x.get("clip_id", ""), x.get("frame", "")))
        self._write_json(description_file, existing)
        return existing

    def _build_video_summary(self, clips):
        summaries = [clip.get("clip_summary", "").strip() for clip in clips if clip.get("clip_summary", "").strip()]
        if not summaries:
            return ""
        joined = " | ".join(summaries[-8:])
        return joined[:1400]

    def _append_memory(self, memory_file, clip_memory_entry):
        memory_data = self._safe_read_json(memory_file, {
            "clips": [],
            "video_summary": ""
        })

        existing_clip_ids = {item.get("clip_id") for item in memory_data.get("clips", [])}
        if clip_memory_entry["clip_id"] not in existing_clip_ids:
            memory_data["clips"].append(clip_memory_entry)

        memory_data["clips"].sort(key=lambda x: x.get("clip_index", 0))
        memory_data["video_summary"] = self._build_video_summary(memory_data["clips"])

        self._write_json(memory_file, memory_data)
        return memory_data

    def _build_batch_prompt(self, clip_info, selected_frames, memory_context):
        frame_lines = []
        for idx, frame in enumerate(selected_frames, start=1):
            frame_lines.append(
                f"{idx}. frame={frame['frame']}, "
                f"clip_id={frame['clip_id']}, "
                f"clip_timestamp={frame['clip_timestamp']:.3f}, "
                f"video_timestamp={frame['video_timestamp']:.3f}"
            )

        frame_manifest = "\n".join(frame_lines)

        if not memory_context:
            memory_context = "No previous video summary available."

        categories_list = ", ".join(Config.GAMING_CATEGORIES)
        signal_keys = ", ".join(Config.SIGNAL_KEYS)
        moment_types = ", ".join(Config.MOMENT_TYPES)

        prompt = f"""
You are analyzing selected key frames from one clip of a competitive gaming video.
**CRITICAL CONTEXT**: This is Valorant gameplay. Look carefully for kill feed updates in the top right, kill banners/skulls at the bottom center, or crosshair hit markers.

Goal:
For every input frame, return a concise gameplay description plus structured category scores and structured visual signals.

Important instructions:
- The frames are in chronological order.
- Use previous video summary only as continuity context.
- Do not hallucinate details not visible in the frames.
- Return valid JSON only.
- Provide exactly one analysis object for every input frame.
- Use the exact input frame filename and exact clip_id.
- Category scores must be numbers between 0.0 and 1.0.
- Signals must be booleans only.
- Keep text concise and retrieval-friendly.
- If a kill is not clearly visible, do not mark it as certain; use lower score.
- If action is unclear, keep scores conservative.

Categories:
{categories_list}

Signals:
{signal_keys}

Allowed action levels:
low, moderate, high

Allowed moment types:
{moment_types}

Previous video summary:
{memory_context}

Current clip metadata:
clip_id: {clip_info['clip_id']}
clip_index: {clip_info['clip_index']}
clip_start_time: {clip_info['start_time']}
clip_end_time: {clip_info['end_time']}

Ordered selected frames:
{frame_manifest}

Return JSON in exactly this structure:
{{
  "frame_analyses": [
    {{
      "frame": "exact filename",
      "clip_id": "exact clip id",
      "clip_timestamp": 0.0,
      "video_timestamp": 0.0,
      "description": "concise gameplay description",
      "scores": {{
        "kill": 0.0,
        "combat": 0.0,
        "clutch": 0.0,
        "healing": 0.0,
        "camping": 0.0,
        "rush": 0.0,
        "objective": 0.0,
        "ability": 0.0
      }},
      "signals": {{
        "enemy_visible": false,
        "firing_visible": false,
        "damage_effect_visible": false,
        "healing_visible": false,
        "crosshair_on_enemy": false,
        "objective_visible": false,
        "ability_visible": false
      }},
      "moment_type": "setup",
      "action_level": "low"
    }}
  ],
  "clip_summary": "one concise sentence summarizing the clip"
}}
"""
        return prompt

    def _clean_json_response(self, text):
        text = text.strip()

        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        return text.strip()

    def _safe_score(self, value):
        try:
            score = float(value)
            if score < 0:
                return 0.0
            if score > 1:
                return 1.0
            return round(score, 3)
        except Exception:
            return 0.0

    def _safe_bool(self, value):
        return bool(value)

    def _safe_action_level(self, value):
        value = (value or "").strip().lower()
        if value in Config.ACTION_LEVEL_MAP:
            return value
        return "low"

    def _safe_moment_type(self, value):
        value = (value or "").strip().lower()
        if value in Config.MOMENT_TYPES:
            return value
        return "transition"

    def describe_frames_batch(self, clip_info, selected_frames, memory_context):
        if not selected_frames:
            return {
                "frame_analyses": [],
                "clip_summary": ""
            }

        prompt = self._build_batch_prompt(clip_info, selected_frames, memory_context)
        contents = [prompt]

        for frame in selected_frames:
            with open(frame["path"], "rb") as f:
                image_bytes = f.read()

            contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg"
                )
            )

        last_error = None

        for key_attempt in range(len(self.fallback_keys) + 1):
            if key_attempt > 0:
                if not self.switch_api_key():
                    break

            for i in range(self.current_model_index, len(self.models)):
                model_id = self.models[i]
                try:
                    print(f"🧠 Batch describing {clip_info['clip_id']} using model: {model_id} with key {self.current_key_index}")

                    response = self.client.models.generate_content(
                        model=model_id,
                        contents=contents
                    )

                    self.current_model_index = i

                    raw_text = response.text.strip()
                    clean_text = self._clean_json_response(raw_text)
                    parsed = json.loads(clean_text)
                    return parsed

                except Exception as e:
                    print(f"⚠️ Error with {model_id}: {e}")
                    last_error = e
                    if self.is_rate_limit_error(e):
                        continue
                    continue

            self.current_model_index = 0

        raise Exception(f"❌ All models and API keys failed for batch description: {last_error}")

    def _aggregate_clip_scores(self, normalized_items):
        category_max_scores = {cat: 0.0 for cat in Config.GAMING_CATEGORIES}
        category_mean_scores = {cat: 0.0 for cat in Config.GAMING_CATEGORIES}
        peak_timestamps = {cat: None for cat in Config.GAMING_CATEGORIES}
        signal_counts = {sig: 0 for sig in Config.SIGNAL_KEYS}
        peak_action_level = "low"

        if not normalized_items:
            return category_max_scores, category_mean_scores, peak_timestamps, signal_counts, peak_action_level

        for item in normalized_items:
            scores = item.get("scores", {})
            signals = item.get("signals", {})

            for cat in Config.GAMING_CATEGORIES:
                score = self._safe_score(scores.get(cat, 0.0))
                category_mean_scores[cat] += score
                if score >= category_max_scores[cat]:
                    category_max_scores[cat] = score
                    peak_timestamps[cat] = item["timestamp"]

            for sig in Config.SIGNAL_KEYS:
                if self._safe_bool(signals.get(sig, False)):
                    signal_counts[sig] += 1

            if Config.ACTION_LEVEL_MAP.get(item.get("action_level", "low"), 1) > Config.ACTION_LEVEL_MAP.get(peak_action_level, 1):
                peak_action_level = item.get("action_level", "low")

        total = len(normalized_items)
        for cat in Config.GAMING_CATEGORIES:
            category_mean_scores[cat] = round(category_mean_scores[cat] / total, 3)

        category_max_scores = {k: round(v, 3) for k, v in category_max_scores.items()}

        return category_max_scores, category_mean_scores, peak_timestamps, signal_counts, peak_action_level

    def process_clip_batch(self, clip_info, selected_frames, description_file, memory_file):
        memory_context, _ = self._load_memory_context(memory_file)

        batch_result = self.describe_frames_batch(
            clip_info=clip_info,
            selected_frames=selected_frames,
            memory_context=memory_context
        )

        frame_analyses = batch_result.get("frame_analyses", [])
        clip_summary = (batch_result.get("clip_summary") or "").strip()

        selected_by_name = {frame["frame"]: frame for frame in selected_frames}
        normalized_items = []

        # Exact filename mapping
        for item in frame_analyses:
            frame_name = (item.get("frame") or "").strip()
            selected_meta = selected_by_name.get(frame_name)
            if not selected_meta:
                continue

            description_text = (item.get("description") or "").strip()
            scores = item.get("scores", {})
            signals = item.get("signals", {})

            normalized_items.append({
                "frame": selected_meta["frame"],
                "clip_id": selected_meta["clip_id"],
                "clip_index": selected_meta["clip_index"],
                "frame_index": selected_meta["frame_index"],
                "clip_timestamp": selected_meta["clip_timestamp"],
                "timestamp": selected_meta["video_timestamp"],
                "video_timestamp": selected_meta["video_timestamp"],
                "description": description_text if description_text else "Gameplay frame with unclear details.",
                "scores": {cat: self._safe_score(scores.get(cat, 0.0)) for cat in Config.GAMING_CATEGORIES},
                "signals": {sig: self._safe_bool(signals.get(sig, False)) for sig in Config.SIGNAL_KEYS},
                "moment_type": self._safe_moment_type(item.get("moment_type")),
                "action_level": self._safe_action_level(item.get("action_level"))
            })

        # Order fallback
        if not normalized_items:
            usable_items = []
            for item in frame_analyses:
                usable_items.append(item)

            pair_count = min(len(selected_frames), len(usable_items))

            for i in range(pair_count):
                selected_meta = selected_frames[i]
                item = usable_items[i]
                scores = item.get("scores", {})
                signals = item.get("signals", {})

                normalized_items.append({
                    "frame": selected_meta["frame"],
                    "clip_id": selected_meta["clip_id"],
                    "clip_index": selected_meta["clip_index"],
                    "frame_index": selected_meta["frame_index"],
                    "clip_timestamp": selected_meta["clip_timestamp"],
                    "timestamp": selected_meta["video_timestamp"],
                    "video_timestamp": selected_meta["video_timestamp"],
                    "description": (item.get("description") or "Gameplay frame with unclear details.").strip(),
                    "scores": {cat: self._safe_score(scores.get(cat, 0.0)) for cat in Config.GAMING_CATEGORIES},
                    "signals": {sig: self._safe_bool(signals.get(sig, False)) for sig in Config.SIGNAL_KEYS},
                    "moment_type": self._safe_moment_type(item.get("moment_type")),
                    "action_level": self._safe_action_level(item.get("action_level"))
                })

        # Safe placeholder fallback
        if not normalized_items:
            for selected_meta in selected_frames:
                normalized_items.append({
                    "frame": selected_meta["frame"],
                    "clip_id": selected_meta["clip_id"],
                    "clip_index": selected_meta["clip_index"],
                    "frame_index": selected_meta["frame_index"],
                    "clip_timestamp": selected_meta["clip_timestamp"],
                    "timestamp": selected_meta["video_timestamp"],
                    "video_timestamp": selected_meta["video_timestamp"],
                    "description": "Gameplay frame with unclear details.",
                    "scores": {cat: 0.0 for cat in Config.GAMING_CATEGORIES},
                    "signals": {sig: False for sig in Config.SIGNAL_KEYS},
                    "moment_type": "transition",
                    "action_level": "low"
                })

        updated_descriptions = self._append_descriptions(description_file, normalized_items)

        category_max_scores, category_mean_scores, peak_timestamps, signal_counts, peak_action_level = self._aggregate_clip_scores(normalized_items)

        memory_entry = {
            "clip_id": clip_info["clip_id"],
            "clip_index": clip_info["clip_index"],
            "start_time": clip_info["start_time"],
            "end_time": clip_info["end_time"],
            "selected_frames_count": len(selected_frames),
            "clip_summary": clip_summary if clip_summary else "Gameplay clip processed.",
            "category_max_scores": category_max_scores,
            "category_mean_scores": category_mean_scores,
            "peak_timestamps": peak_timestamps,
            "signal_counts": signal_counts,
            "peak_action_level": peak_action_level
        }

        updated_memory = self._append_memory(memory_file, memory_entry)

        time.sleep(self.request_delay)

        return {
            "frame_analyses": normalized_items,
            "clip_summary": clip_summary,
            "updated_descriptions": updated_descriptions,
            "updated_memory": updated_memory
        }