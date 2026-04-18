import json
import time
import pandas as pd
from google import genai
from config import Config


class HighlightMatcher:
    def __init__(self):
        self.api_key = Config.API_KEY
        self.model_id = Config.MATCHER_MODEL
        self.client = genai.Client(api_key=self.api_key)
        self.max_retries = Config.MATCHER_MAX_RETRIES
        self.retry_wait_seconds = Config.MATCHER_RETRY_WAIT_SECONDS

    def _is_rate_limit_error(self, e):
        msg = str(e).lower()
        return (
            "429" in msg
            or "quota" in msg
            or "resource_exhausted" in msg
            or "rate" in msg
            or "limit" in msg
        )

    def _call_model_with_retry(self, prompt):
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"🧠 Matching with {self.model_id} | attempt {attempt}/{self.max_retries}")

                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt
                )

                return response.text or ""

            except Exception as e:
                last_error = e

                if self._is_rate_limit_error(e) and attempt < self.max_retries:
                    print(f"⏳ Matcher rate limit hit. Waiting {self.retry_wait_seconds} seconds before retry...")
                    time.sleep(self.retry_wait_seconds)
                    continue

                raise last_error

        raise last_error

    def match_highlights(self, descriptions_file, memory_file, user_prompt, output_file):
        """
        Match user prompt with frame descriptions using cumulative memory context.
        Non-chunked version, but with better model + retry.
        """
        with open(descriptions_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        memory_context = ""
        if memory_file and memory_file.strip():
            try:
                with open(memory_file, "r", encoding="utf-8") as f:
                    memory_data = json.load(f)
                    memory_context = memory_data.get("cumulative_summary", "")
            except Exception:
                memory_context = ""

        # Optional protection: trim very long memory
        if len(memory_context) > 3000:
            memory_context = memory_context[-3000:]

        frames_text = ""
        for item in data:
            frames_text += f"""
Frame: {item['frame']}
Clip ID: {item.get('clip_id', '')}
Timestamp: {item['timestamp']}
Description: {item['description']}

"""

        analysis_prompt = f"""
A user wants highlight moments from a video.

User request:
{user_prompt}

Global video memory summary:
{memory_context if memory_context else "No global memory summary available."}

Below are selected frame descriptions with timestamps.

Task:
- Select only timestamps highly relevant to the user request.
- Use the memory summary only as global context.
- Base final timestamp selection on frame descriptions.
- Return only the most relevant moments.
- Do not invent timestamps.

Return results in this exact format:
timestamp, description

Only include highly relevant lines.

Frames:
{frames_text}
"""

        result_text = self._call_model_with_retry(analysis_prompt)

        rows = []
        seen = set()

        for line in result_text.split("\n"):
            line = line.strip()
            if "," in line and not line.lower().startswith("timestamp"):
                parts = line.split(",", 1)
                try:
                    ts = float(parts[0].strip())
                    desc = parts[1].strip()
                    key = (ts, desc)
                    if key not in seen:
                        rows.append({
                            "timestamp": ts,
                            "description": desc
                        })
                        seen.add(key)
                except ValueError:
                    continue

        df = pd.DataFrame(rows)
        df.to_excel(output_file, index=False)

        return rows