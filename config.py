import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

    # Paths
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    FRAMES_FOLDER = os.path.join(BASE_DIR, 'frames')
    OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
    CLIPS_FOLDER = os.path.join(BASE_DIR, 'clips')

    for folder in [UPLOAD_FOLDER, FRAMES_FOLDER, OUTPUT_FOLDER, CLIPS_FOLDER]:
        os.makedirs(folder, exist_ok=True)

    # Video settings
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

    # Clip settings
    CLIP_DURATION = 5  # seconds

    # Frame extraction
    HIGH_FPS = 5

    # Filtering thresholds
    FRAME_DIFF_THRESHOLD = 0.35
    SSIM_DIFF_THRESHOLD = 0.30

    # Filtering behavior
    KEEP_FIRST_FRAME = True
    KEEP_LAST_FRAME = True
    PRE_CHANGE_FRAMES = 1
    MIN_SELECTED_PER_CLIP = 5
    MAX_SELECTED_PER_CLIP = 5

    # API settings
    API_KEY = os.getenv("GOOGLE_API_KEY")
    FALLBACK_API_KEYS = [
        os.getenv("GOOGLE_API_KEY_2")
    ]

    # Description generation models (fallback order)
    MODELS = [
        "gemma-3-27b-it",
        "gemma-3-12b-it",
        "gemma-3-4b-it"
    ]

    # Prompt enhancer model
    PROMPT_ENHANCER_MODEL = "gemini-2.5-flash-lite"

    # Matcher model
    MATCHER_MODEL = "gemini-2.5-flash"

    # Retry settings
    REQUEST_DELAY = 1
    MATCHER_MAX_RETRIES = 3
    MATCHER_RETRY_WAIT_SECONDS = 35

    # Video processing
    PRE_CONTEXT = 2
    POST_CONTEXT = 3
    MERGE_THRESHOLD = 4

    # Output settings
    OUTPUT_VIDEO_NAME = "final_highlights.mp4"

    # Output file names
    DESCRIPTION_JSON_NAME = "description.json"
    MEMORY_JSON_NAME = "memory.json"
    HIGHLIGHTS_XLSX_NAME = "highlights.xlsx"