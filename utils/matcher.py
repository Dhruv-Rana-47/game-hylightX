import json
import re
import pandas as pd
from config import Config


class HighlightMatcher:
    """
    Rule-based matcher for gaming categories.
    Categories are primary. Optional text prompt is a refinement layer only.
    """

    def __init__(self):
        self.thresholds = Config.CATEGORY_THRESHOLDS
        self.required_signals = Config.REQUIRED_SIGNALS
        self.action_level_map = Config.ACTION_LEVEL_MAP

    def _safe_read_json(self, file_path, default_value):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default_value

    def _tokenize(self, text):
        if not text:
            return set()
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        stop = {
            "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "with",
            "show", "me", "only", "all", "moment", "moments", "highlight", "highlights"
        }
        return {t for t in tokens if t not in stop and len(t) > 2}

    def _text_boost(self, refined_prompt, clip_summary, frame_descriptions):
        query_tokens = self._tokenize(refined_prompt)
        if not query_tokens:
            return 0.0

        clip_tokens = self._tokenize(clip_summary)
        frame_tokens = self._tokenize(" ".join(frame_descriptions))
        all_tokens = clip_tokens.union(frame_tokens)

        if not all_tokens:
            return 0.0

        overlap = len(query_tokens.intersection(all_tokens))
        boost = min(0.15, overlap * 0.03)
        return round(boost, 3)

    def _has_required_signals(self, clip_entry, category):
        required = self.required_signals.get(category, [])
        signal_counts = clip_entry.get("signal_counts", {})
        for sig in required:
            if signal_counts.get(sig, 0) <= 0:
                return False
        return True

    def _category_match(self, clip_entry, category):
        max_scores = clip_entry.get("category_max_scores", {})
        mean_scores = clip_entry.get("category_mean_scores", {})
        peak_action_level = clip_entry.get("peak_action_level", "low")

        max_score = float(max_scores.get(category, 0.0))
        mean_score = float(mean_scores.get(category, 0.0))
        action_rank = self.action_level_map.get(peak_action_level, 1)

        threshold = self.thresholds.get(category, 0.7)
        signals_ok = self._has_required_signals(clip_entry, category)

        if category == "kill":
            return max_score >= threshold
        if category == "combat":
            return max_score >= threshold and signals_ok and action_rank >= 2
        if category == "clutch":
            return max_score >= threshold and signals_ok and action_rank >= 2
        if category == "healing":
            return max_score >= threshold and signals_ok
        if category == "camping":
            return max_score >= threshold and mean_score >= 0.45
        if category == "rush":
            return max_score >= threshold and action_rank >= 2
        if category == "objective":
            return max_score >= threshold and signals_ok
        if category == "ability":
            return max_score >= threshold and signals_ok

        return False

    def match_highlights(self, descriptions_file, memory_file, selected_categories, optional_prompt, refined_prompt, output_file):
        descriptions = self._safe_read_json(descriptions_file, [])
        memory = self._safe_read_json(memory_file, {"clips": [], "video_summary": ""})

        descriptions_by_clip = {}
        for item in descriptions:
            descriptions_by_clip.setdefault(item.get("clip_id", ""), []).append(item)

        rows = []
        seen = set()

        for clip_entry in memory.get("clips", []):
            clip_id = clip_entry.get("clip_id")
            clip_frames = descriptions_by_clip.get(clip_id, [])
            clip_summary = clip_entry.get("clip_summary", "")
            peak_timestamps = clip_entry.get("peak_timestamps", {})
            max_scores = clip_entry.get("category_max_scores", {})

            frame_descs = [frame.get("description", "") for frame in clip_frames]
            text_boost = self._text_boost(refined_prompt or optional_prompt, clip_summary, frame_descs)

            for category in selected_categories:
                if not self._category_match(clip_entry, category):
                    continue

                ts = peak_timestamps.get(category)
                if ts is None:
                    ts = clip_entry.get("start_time", 0.0)

                score = float(max_scores.get(category, 0.0)) + text_boost
                score = round(min(score, 1.0), 3)

                description = f"{Config.CATEGORY_LABELS.get(category, category)} | score={score} | {clip_summary}"

                key = (round(float(ts), 2), category)
                if key in seen:
                    continue

                rows.append({
                    "timestamp": round(float(ts), 3),
                    "description": description,
                    "category": category,
                    "score": score
                })
                seen.add(key)

        rows.sort(key=lambda x: (x["timestamp"], -x["score"]))

        df = pd.DataFrame(rows)
        df.to_excel(output_file, index=False)

        return rows