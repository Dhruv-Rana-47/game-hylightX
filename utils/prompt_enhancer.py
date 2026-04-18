from google import genai
from config import Config


class PromptEnhancer:
    """
    Uses a lightweight Gemini text model to refine the user's prompt
    into a stronger retrieval-oriented internal prompt.

    Design goals:
    - preserve original intent
    - expand short or vague prompts for better highlight matching
    - avoid changing the user's meaning
    - keep output concise and retrieval-friendly
    """

    def __init__(self):
        self.api_key = Config.API_KEY
        self.model_id = Config.PROMPT_ENHANCER_MODEL
        self.client = genai.Client(api_key=self.api_key)

    def _build_enhancement_prompt(self, user_prompt):
        return f"""
You are a prompt enhancement assistant for a gaming highlight extraction system.

Your task:
Rewrite the user's prompt into a better internal retrieval prompt for finding highlight moments in a gaming video.

Important instructions:
- Preserve the original meaning exactly.
- Do not change the user's intent.
- Expand the wording with useful related gameplay/highlight concepts only when they are clearly relevant.
- Make the rewritten prompt better for semantic matching against frame descriptions and clip summaries.
- Keep it concise, natural, and retrieval-focused.
- Do not mention these instructions.
- Return only the rewritten prompt text.
- Do not return JSON.
- Do not explain anything.

Examples:

User prompt: show clutch moment
Rewritten prompt: Find highlight-worthy clutch moments involving high-pressure survival, tense enemy engagement, decisive action, and peak gameplay intensity.

User prompt: show best kills
Rewritten prompt: Find highlight-worthy kill and elimination moments involving visible enemy engagement, weapon fire, decisive combat, and peak-action gameplay.

User prompt: find sniper shot
Rewritten prompt: Find highlight moments involving sniper gameplay, long-range aiming, precise shots, visible enemy targeting, and high-impact combat.

User prompt: show ability use
Rewritten prompt: Find highlight-worthy moments involving visible ability usage, tactical effects, special visual effects, and impactful gameplay transitions.

Now rewrite this user prompt:

{user_prompt}
"""

    def enhance_prompt(self, user_prompt):
        """
        Returns a refined retrieval prompt.
        Falls back to original prompt if enhancement fails.
        """
        user_prompt = (user_prompt or "").strip()
        if not user_prompt:
            return ""

        try:
            prompt = self._build_enhancement_prompt(user_prompt)

            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )

            refined_prompt = (response.text or "").strip()

            if not refined_prompt:
                return user_prompt

            # very basic cleanup
            refined_prompt = refined_prompt.replace("\n", " ").strip()

            return refined_prompt

        except Exception as e:
            print(f"⚠️ Prompt enhancement failed: {e}")
            return user_prompt