# HylightX: AI-Powered Gaming Highlight Extractor
## Project Technical Report

### 1. Project Overview
HylightX is an advanced video processing platform designed to automatically extract gameplay highlights (kills, multi-kills, etc.) from raw gaming footage. It leverages Vision LLMs (Gemma/Gemini) to "watch" the video and identify significant moments based on visual cues.

---

### 2. Core Technology Stack
- **Backend**: Flask (Python)
- **Database**: SQLite with SQLAlchemy ORM
- **Authentication**: Flask-Login & Authlib (Google OAuth 2.0)
- **AI Models**: 
    - `Gemma 3` (Vision-Language model) for frame analysis.
    - `Gemini 2.5 Flash` (Fallback/Enhancer) for prompt optimization.
- **Video Processing**: FFmpeg & MoviePy
- **Computer Vision**: OpenCV (Frame extraction & filtering)

---

### 3. Technical Workflow (Step-by-Step)

#### Phase 1: Authentication & Security
- Users can register/login via standard email/password or use **Google OAuth**.
- The database stores hashed passwords (using Werkzeug) or handles password-less entries for Google users.
- All processing routes are protected by `@login_required` decorators.

#### Phase 2: Video Pre-processing
1. **Segmentation**: The uploaded video is split into **10-second clips** to balance processing speed and API quota management.
2. **Optimized Frame Extraction**:
   - Frames are extracted at **5 FPS**.
   - To reduce latency and token costs, all extracted frames are automatically **resized to 720p** before being sent to the AI.
3. **Adaptive Filtering**:
   - Uses **SSIM (Structural Similarity Index)** and **MSE (Mean Squared Error)** to detect significant visual changes.
   - Discards redundant/static frames, keeping only 3-10 high-action frames per clip.

#### Phase 3: AI Analysis (The "Brain")
1. **Context-Aware Prompts**: The system uses a specialized **Valorant-tuned prompt** that instructs the AI to look for specific visual signals like the "Kill Banner," "Kill Feed" notifications, and "Skull" icons.
2. **Batch Processing**: Multiple frames are sent in a single batch request to the LLM (Gemma 3) to minimize API call counts.
3. **Memory System**: The AI maintains a "short-term memory" of what happened in previous clips to detect continuous multi-kills.

#### Phase 4: Highlight Matching & Extraction
1. **Refined Prompting**: A secondary LLM (Prompt Enhancer) refines the user's search query based on the overall video context.
2. **Timestamp Detection**: The `HighlightMatcher` scans the AI-generated descriptions and extracts precise start/end times where the action occurred.
3. **Final Render**: FFmpeg/MoviePy cuts the raw video based on these timestamps, adds a **2s pre-context** and **3s post-context** for better flow, and merges them into a final `final_highlights.mp4`.

---

### 4. Recent Optimization Updates (v3.2)
- **Speed Increase**: Moved from 5s clips to 10s clips, reducing the number of API requests by 50%.
- **Quota Resilience**: Implemented a `REQUEST_DELAY` (4.2s) and adaptive frame sampling to stay within the Google AI Free Tier rate limits (30 RPM / 15k TPM).
- **Quality Control**: Relaxed matching logic to ensure "kills" are captured even if visual signals are slightly obscured.

---

### 5. Project Directory Structure
```text
game_hylightX/
├── app.py              # Main Flask application & routes
├── config.py           # Central configuration & AI thresholds
├── models.py           # Database models (User)
├── extensions.py       # Flask extensions initialization
├── instance/           # Database & local storage
├── templates/          # HTML5/CSS3 Frontend files
├── utils/              # Core logic modules
│   ├── video_segmenter.py
│   ├── frame_extractor.py
│   ├── frame_filter.py
│   ├── description_generator.py
│   ├── matcher.py
│   └── video_processor.py
└── .env                # API Keys & Secrets (Private)
```

---

### 6. Security Implementation
- **Secret Management**: All sensitive keys (Google API, OAuth Secrets) are stored in environment variables.
- **CSRF Protection**: State/Nonce tokens are used during the Google OAuth flow to prevent interception.
- **Data Privacy**: Local database ensures user data is handled securely within the environment.
