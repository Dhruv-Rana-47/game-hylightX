======= gaming video system prompt=================
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
============== End of gaming prmopt ================


================  general emotions extraction prompt ===========
     prompt = f"""
You are analyzing selected key frames from one video clip for emotion recognition benchmarking.

Primary task:
Describe each frame so that later retrieval can accurately find clips showing specific emotions such as happiness, sadness, anger, fear, surprise, disgust, or neutral expression.

Strict rules:
- The frames are in chronological order.
- Use previous memory only as continuity context.
- Do not hallucinate details not visible in the frames.
- Return valid JSON only.
- Provide one output description for every input frame.
- The number of output frame_descriptions must exactly match the number of input frames.
- Use the exact frame filename and exact clip_id provided in the input.
- Focus first on the main visible subject's facial expression and emotional cues.
- Then consider body language and gesture.
- Background details should be brief unless they help interpret the emotion.
- If the emotion is ambiguous, explicitly say "emotion unclear" instead of guessing.

Emotion labels to consider when visually supported:
- happiness
- sadness
- anger
- fear
- surprise
- disgust
- neutral

For each frame description:
- identify the most likely visible emotion if possible
- mention visible evidence for that emotion
- keep the description concise but specific
- avoid generic wording like "person standing" unless nothing else is visible

Good description examples:
- "The person appears surprised, with wide eyes, raised eyebrows, and an open mouth."
- "The person shows sadness, with downturned lips, heavy eyes, and a subdued expression."
- "The person appears angry, with a tense face, narrowed eyes, and a strong frown."
- "The person shows disgust, with a wrinkled nose and tightened facial expression."
- "The face is partially visible and emotion is unclear."

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
    "description": "emotion-focused visual description with visible cues"
    }}
],
"clip_summary": "short summary of what happens in this clip, including visible emotional state or emotional transition if present",
"cumulative_summary": "updated running summary of the video so far, including emotional continuity where clearly visible"
}}
"""