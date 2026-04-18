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
            json.dump(data, f, indent=4, ensure_ascii=False)
        os.replace(temp_path, file_path)

    def _load_memory_context(self, memory_file):
        memory_data = self._safe_read_json(memory_file, {
            "clips": [],
            "cumulative_summary": ""
        })
        return memory_data.get("cumulative_summary", "").strip(), memory_data

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

    def _append_memory(self, memory_file, clip_memory_entry):
        memory_data = self._safe_read_json(memory_file, {
            "clips": [],
            "cumulative_summary": ""
        })

        existing_clip_ids = {item.get("clip_id") for item in memory_data.get("clips", [])}

        if clip_memory_entry["clip_id"] not in existing_clip_ids:
            memory_data["clips"].append(clip_memory_entry)

        memory_data["clips"].sort(key=lambda x: x.get("clip_index", 0))
        memory_data["cumulative_summary"] = clip_memory_entry.get("cumulative_summary", memory_data.get("cumulative_summary", ""))

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
            memory_context = "No previous memory available. This may be the beginning of the video."

        prompt = f"""
You are analyzing selected key frames from one clip of a competitive shooter gameplay video, such as Valorant.

Primary goal:
Generate frame descriptions that help downstream retrieval identify highlight-worthy moments such as fights, kills, clutch plays, enemy encounters, and intense action sequences.

Strict instructions:
- The frames are in chronological order.
- Use previous memory only as continuity context.
- Do not hallucinate details not visible in the frames.
- Return valid JSON only.
- Provide one output description for every input frame.
- The number of output frame_descriptions must exactly match the number of input frames.
- Use the exact frame filename and exact clip_id provided in the input.
- Focus on visible gameplay action, engagement level, and highlight potential.
- If a kill or elimination is not clearly visible, describe it as a likely combat or engagement moment rather than stating it as fact.
- If a frame is low action, say so.

Pay special attention to these visual signals:
- enemy visible on screen
- crosshair aligned toward an opponent
- firing or muzzle flash
- damage/combat effects
- explosion, smoke, flash, or ability effect
- sudden aggressive movement
- close-range encounter
- multi-enemy pressure
- tense angle holding
- objective pressure
- visually intense or decisive moment
- aftermath of a fight

For each frame description:
- state what is visibly happening
- mention the action intensity: low, moderate, or high if visually inferable
- mention whether the frame appears to be setup, engagement, peak-action, or post-fight
- mention visible combat cues and highlight relevance
- avoid generic phrases unless the frame truly has little information

Previous cumulative memory:
{memory_context}

Current clip metadata:
clip_id: {clip_info['clip_id']}
clip_index: {clip_info['clip_index']}
clip_start_time: {clip_info['start_time']}
clip_end_time: {clip_info['end_time']}

Ordered selected frames:
{frame_manifest}

Return JSON in this exact structure:
{{
  "frame_descriptions": [
    {{
      "frame": "exact filename",
      "clip_id": "exact clip id",
      "clip_timestamp": 0.0,
      "video_timestamp": 0.0,
      "description": "gameplay-focused description emphasizing visible action, combat cues, and highlight intensity"
    }}
  ],
  "clip_summary": "short summary of the clip including action buildup, engagement, and highlight-worthy progression if present",
  "cumulative_summary": "updated running summary of the gameplay so far, including combat flow and major action continuity if visible"
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

    def describe_frames_batch(self, clip_info, selected_frames, memory_context):
        """
        One model request per clip.
        """
        if not selected_frames:
            return {
                "frame_descriptions": [],
                "clip_summary": "",
                "cumulative_summary": memory_context or ""
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
                    print(f"📦 Raw model response for {clip_info['clip_id']}:\n{clean_text}\n")
                    
                    parsed = json.loads(clean_text)

                    return parsed

                except Exception as e:
                    print(f"⚠️ Error with {model_id}: {e}")
                    if self.is_rate_limit_error(e):
                        last_error = e
                        continue
                    else:
                        last_error = e
                        continue

            self.current_model_index = 0

        raise Exception(f"❌ All models and API keys failed for batch description: {last_error}")
    
    def process_clip_batch(self, clip_info, selected_frames, description_file, memory_file):
        print("🔥 NEW process_clip_batch loaded")
        """
        End-to-end processing for one clip:
        - load prior memory
        - batch describe selected frames
        - append description.json
        - append memory.json

        Robust version:
        - tries exact filename matching first
        - falls back to index-based mapping if model filenames are not exact
        - never crashes just because mapping is weak
        """
        memory_context, _ = self._load_memory_context(memory_file)

        batch_result = self.describe_frames_batch(
            clip_info=clip_info,
            selected_frames=selected_frames,
            memory_context=memory_context
        )

        frame_descriptions = batch_result.get("frame_descriptions", [])
        clip_summary = (batch_result.get("clip_summary") or "").strip()
        cumulative_summary = (batch_result.get("cumulative_summary") or "").strip()

        print(f"📌 {clip_info['clip_id']} returned {len(frame_descriptions)} frame_descriptions")

        selected_by_name = {frame["frame"]: frame for frame in selected_frames}
        normalized_items = []

        # PASS 1: exact filename matching
        for item in frame_descriptions:
            frame_name = (item.get("frame") or "").strip()
            selected_meta = selected_by_name.get(frame_name)

            if not selected_meta:
                continue

            description_text = (item.get("description") or "").strip()
            if not description_text:
                continue

            normalized_items.append({
                "frame": selected_meta["frame"],
                "clip_id": selected_meta["clip_id"],
                "clip_index": selected_meta["clip_index"],
                "frame_index": selected_meta["frame_index"],
                "clip_timestamp": selected_meta["clip_timestamp"],
                "timestamp": selected_meta["video_timestamp"],
                "video_timestamp": selected_meta["video_timestamp"],
                "description": description_text
            })

        # PASS 2: fallback to order-based mapping
        if not normalized_items:
            print(f"⚠️ Exact frame-name matching failed for {clip_info['clip_id']}. Using order-based fallback.")

            usable_items = []
            for item in frame_descriptions:
                description_text = (item.get("description") or "").strip()
                if description_text:
                    usable_items.append(item)

            pair_count = min(len(selected_frames), len(usable_items))

            for i in range(pair_count):
                selected_meta = selected_frames[i]
                item = usable_items[i]
                description_text = (item.get("description") or "").strip()

                if not description_text:
                    continue

                normalized_items.append({
                    "frame": selected_meta["frame"],
                    "clip_id": selected_meta["clip_id"],
                    "clip_index": selected_meta["clip_index"],
                    "frame_index": selected_meta["frame_index"],
                    "clip_timestamp": selected_meta["clip_timestamp"],
                    "timestamp": selected_meta["video_timestamp"],
                    "video_timestamp": selected_meta["video_timestamp"],
                    "description": description_text
                })

        # PASS 3: safe placeholder fallback
        if not normalized_items:
            print(f"⚠️ No usable structured mapping for {clip_info['clip_id']}. Creating placeholder descriptions.")

            for selected_meta in selected_frames:
                normalized_items.append({
                    "frame": selected_meta["frame"],
                    "clip_id": selected_meta["clip_id"],
                    "clip_index": selected_meta["clip_index"],
                    "frame_index": selected_meta["frame_index"],
                    "clip_timestamp": selected_meta["clip_timestamp"],
                    "timestamp": selected_meta["video_timestamp"],
                    "video_timestamp": selected_meta["video_timestamp"],
                    "description": "Gameplay frame selected from this clip, but the model did not return a usable structured frame description."
                })

        print(f"✅ {clip_info['clip_id']} normalized_items count = {len(normalized_items)}")

        updated_descriptions = self._append_descriptions(description_file, normalized_items)

        memory_entry = {
            "clip_id": clip_info["clip_id"],
            "clip_index": clip_info["clip_index"],
            "start_time": clip_info["start_time"],
            "end_time": clip_info["end_time"],
            "selected_frames_count": len(selected_frames),
            "clip_summary": clip_summary if clip_summary else "Clip processed but summary was weak or missing.",
            "cumulative_summary": cumulative_summary if cumulative_summary else memory_context
        }

        updated_memory = self._append_memory(memory_file, memory_entry)

        time.sleep(self.request_delay)

        return {
            "frame_descriptions": normalized_items,
            "clip_summary": clip_summary,
            "cumulative_summary": cumulative_summary,
            "updated_descriptions": updated_descriptions,
            "updated_memory": updated_memory
        }
