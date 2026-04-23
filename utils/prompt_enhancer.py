from google import genai
from config import Config


class PromptEnhancer:
    """
    Refines an optional free-text note from the user.
    Categories remain primary; text is only a refinement layer.
    """

    def __init__(self):
        self.api_key = Config.API_KEY
        self.model_id = Config.PROMPT_ENHANCER_MODEL
        self.client = genai.Client(api_key=self.api_key)

    def _build_enhancement_prompt(self, user_prompt, selected_categories, video_summary):
        categories_text = ", ".join(selected_categories) if selected_categories else "none"

        return f"""
You are refining a user's optional note for a gaming highlight extraction system.

The system already has selected highlight categories:
{categories_text}

Your job:
Rewrite the user's note into a concise refinement prompt that helps rank matching clips more accurately.

Rules:
- Preserve the original meaning.
- Do not replace or contradict the selected categories.
- Keep it short and retrieval-friendly.
- Use the video summary as context.
- Return only the refined text.
- Do not return JSON.
- Do not explain anything.

User note:
{user_prompt}

Video summary:
{video_summary if video_summary else "No summary available."}
"""

    def enhance_prompt(self, user_prompt, selected_categories, video_summary):
        user_prompt = (user_prompt or "").strip()
        if not user_prompt:
            return ""

        try:
            prompt = self._build_enhancement_prompt(user_prompt, selected_categories, video_summary)
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            refined_prompt = (response.text or "").strip()
            return refined_prompt.replace("\n", " ").strip() if refined_prompt else user_prompt
        except Exception as e:
            print(f"⚠️ Prompt enhancement failed: {e}")
            return user_prompt