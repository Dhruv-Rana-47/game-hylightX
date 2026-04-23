import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-secret-key-change-in-production"

    # Database settings
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL") or "sqlite:///app.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Paths
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    FRAMES_FOLDER = os.path.join(BASE_DIR, "frames")
    OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
    CLIPS_FOLDER = os.path.join(BASE_DIR, "clips")

    for folder in [UPLOAD_FOLDER, FRAMES_FOLDER, OUTPUT_FOLDER, CLIPS_FOLDER]:
        os.makedirs(folder, exist_ok=True)

    # Video settings
    ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
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
    MAX_SELECTED_PER_CLIP = 7

    # API settings
    API_KEY = os.getenv("GOOGLE_API_KEY")
    FALLBACK_API_KEYS = [
        os.getenv("GOOGLE_API_KEY_2")
    ]

    # Vision models for description stage
    MODELS = [
        "gemma-3-27b-it",
        "gemma-3-12b-it",
        "gemma-3-4b-it"
    ]

    # Optional prompt enhancement model
    PROMPT_ENHANCER_MODEL = "gemini-2.5-flash-lite"

    # These are no longer used for LLM matching, but kept for compatibility
    MATCHER_MODEL = "gemini-2.5-flash"
    MATCHER_MAX_RETRIES = 3
    MATCHER_RETRY_WAIT_SECONDS = 35

    REQUEST_DELAY = 2.5  # seconds

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

    # Category-driven gaming setup
    GAMING_CATEGORIES = [
        "kill",
        "combat",
        "clutch",
        "healing",
        "camping",
        "rush",
        "objective",
        "ability"
    ]

    CATEGORY_LABELS = {
        "kill": "Kill",
        "combat": "Combat",
        "clutch": "Clutch",
        "healing": "Healing",
        "camping": "Camping",
        "rush": "Rush",
        "objective": "Bomb / Objective",
        "ability": "Ability"
    }

    CATEGORY_THRESHOLDS = {
        "kill": 0.65,
        "combat": 0.68,
        "clutch": 0.70,
        "healing": 0.72,
        "camping": 0.68,
        "rush": 0.68,
        "objective": 0.70,
        "ability": 0.68
    }

    ACTION_LEVEL_MAP = {
        "low": 1,
        "moderate": 2,
        "high": 3
    }

    REQUIRED_SIGNALS = {
        "kill": [],
        "combat": ["enemy_visible"],
        "clutch": ["enemy_visible"],
        "healing": ["healing_visible"],
        "camping": [],
        "rush": [],
        "objective": ["objective_visible"],
        "ability": ["ability_visible"]
    }

    SIGNAL_KEYS = [
        "enemy_visible",
        "firing_visible",
        "damage_effect_visible",
        "healing_visible",
        "crosshair_on_enemy",
        "objective_visible",
        "ability_visible"
    ]

    MOMENT_TYPES = [
        "setup",
        "engagement",
        "peak-action",
        "post-fight",
        "transition"
    ]