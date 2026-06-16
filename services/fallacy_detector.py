from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

FALLACY_PROMPT = """You are a strict logic professor who understands the difference between legitimate reasoning and actual logical fallacies.

Argument: {argument}

Analyze ONLY for REAL logical fallacies. Be very conservative — only flag something if it is clearly and unambiguously a fallacy.

CRITICAL RULES before flagging:
- Citing peer-reviewed studies, academic research, or scientific meta-analyses is NOT appeal to authority. That is legitimate evidence.
- Appeal to authority ONLY applies when someone cites a person OUTSIDE their domain of expertise (e.g. asking an actor about vaccines).
- Mentioning real documented harms (e.g. Facebook's leaked internal research) is NOT appeal to emotion. It is factual evidence.
- Appeal to emotion ONLY applies when someone uses purely emotional language with zero factual basis (e.g. "think of the children!" with no data).
- Summarizing the opponent's position briefly to counter it is NOT a strawman. Strawman ONLY applies when someone deliberately distorts the opponent's argument into something they never said.
- Using strong confident language is NOT a fallacy.
- Making causal claims supported by longitudinal data is NOT a fallacy.

Check ONLY for these — and only flag if you are very confident:
- strawman: deliberately misrepresenting what the opponent actually said
- ad hominem: attacking the person's character instead of their argument
- false dichotomy: presenting only two options when clearly more exist
- slippery slope: claiming one event inevitably leads to extreme consequences with zero evidence
- hasty generalization: drawing a sweeping conclusion from one or two isolated examples
- circular reasoning: using the conclusion itself as the main premise

Return ONLY this JSON, no extra text:
{{
  "fallacies_found": true or false,
  "fallacies": [
    {{
      "name": "fallacy name",
      "explanation": "one sentence explaining exactly how they used it"
    }}
  ],
  "clean_argument": true or false
}}

If no clear fallacy exists, return fallacies_found: false and empty array. When in doubt, do NOT flag it.
"""


class FallacyDetector:
    def detect(self, argument: str) -> dict:
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{
                    "role": "user",
                    "content": FALLACY_PROMPT.format(argument=argument)
                }],
                temperature=0.2,
                max_tokens=500
            )

            raw = response.choices[0].message.content.strip()
            # Strip markdown code blocks if model wraps in them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())

        except Exception as e:
            print(f"Fallacy detection error: {e}")
            return {"fallacies_found": False, "fallacies": [], "clean_argument": True}