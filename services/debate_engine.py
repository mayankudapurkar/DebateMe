from groq import Groq
from services.rag_service import RAGService
from services.fallacy_detector import FallacyDetector
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a razor-sharp, relentless debate opponent. Your sole job is to DESTROY the user's argument.

Rules:
- Always argue the OPPOSITE of whatever the user claims. Never agree, never concede fully.
- Every counter-argument MUST be grounded in the EVIDENCE provided. Don't make things up.
- Be aggressive, confident, and direct. No fluff, no filler.
- Structure your response as:
  1. One sentence directly attacking their core claim
  2. Evidence-backed counter (cite the source if possible)
  3. One sharp closing point that puts them on the defensive
- Keep it under 150 words. Punchy beats verbose every time.
- Never repeat a counter-argument you've already made in this debate.
- If they make a strong point, say "Fair point — but..." then pivot to a stronger counter.

Evidence to use:
{evidence}

Arguments you've already made (DO NOT repeat these):
{used_arguments}
"""

SCORING_PROMPT = """You are an impartial debate judge. Analyze this debate and score both sides.

Topic: {topic}
User's position: {user_position}
AI's position: opposite of user

Debate transcript:
{transcript}

Score each side (0-10) on:
1. Argument strength
2. Evidence usage
3. Logical consistency
4. Handling of opponent's points

Return a JSON with:
{{
  "user_score": {{
    "argument_strength": int,
    "evidence_usage": int,
    "logical_consistency": int,
    "handling_opponent": int,
    "total": int,
    "verdict": "string (1-2 sentences)"
  }},
  "ai_score": {{
    "argument_strength": int,
    "evidence_usage": int,
    "logical_consistency": int,
    "handling_opponent": int,
    "total": int,
    "verdict": "string (1-2 sentences)"
  }},
  "winner": "user or ai or draw",
  "summary": "string (2-3 sentences summarizing the debate)"
}}
"""


class DebateEngine:
    def __init__(self):
        self.rag = RAGService()
        self.fallacy_detector = FallacyDetector()
        self.conversation_history = []
        self.used_arguments = []  # tracks AI's past counters
        self.detected_fallacies = []
        self.topic = None
        self.user_position = None

    def start_debate(self, topic: str, user_position: str) -> dict:
        self.topic = topic
        self.user_position = user_position
        self.conversation_history = []
        self.used_arguments = []

        chunks_indexed = self.rag.fetch_and_index(topic)

        evidence = self.rag.retrieve_counter_evidence(user_position)
        evidence_text = "\n\n".join(evidence)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(
                evidence=evidence_text,
                used_arguments="None yet."
            )},
            {"role": "user", "content": f"I believe: {user_position}"}
        ]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.8,
            max_tokens=300
        )

        ai_reply = response.choices[0].message.content
        self.used_arguments.append(ai_reply)

        self.conversation_history.append({"role": "user", "content": user_position})
        self.conversation_history.append({"role": "assistant", "content": ai_reply})

        return {
            "ai_response": ai_reply,
            "chunks_indexed": chunks_indexed,
            "turn": 1,
            "fallacies": None  # no fallacy check on opening statement
        }

    def continue_debate(self, user_argument: str) -> dict:
        if not self.topic:
            raise ValueError("No active debate. Start one first.")

        # Detect fallacies in user's argument
        fallacy_result = self.fallacy_detector.detect(user_argument)

        if fallacy_result.get("fallacies_found"):
            self.detected_fallacies.extend(fallacy_result["fallacies"])

        # Get fresh counter-evidence
        evidence = self.rag.retrieve_counter_evidence(user_argument)
        evidence_text = "\n\n".join(evidence)

        # List of AI's past counters so it doesn't repeat
        used_text = "\n".join([f"- {a[:100]}..." for a in self.used_arguments]) or "None yet."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(
                evidence=evidence_text,
                used_arguments=used_text
            )},
        ] + self.conversation_history + [
            {"role": "user", "content": user_argument}
        ]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.8,
            max_tokens=300
        )

        ai_reply = response.choices[0].message.content
        self.used_arguments.append(ai_reply)

        self.conversation_history.append({"role": "user", "content": user_argument})
        self.conversation_history.append({"role": "assistant", "content": ai_reply})

        return {
            "ai_response": ai_reply,
            "turn": len(self.conversation_history) // 2,
            "fallacies": fallacy_result
        }

    def end_debate(self) -> dict:
        if not self.conversation_history:
            raise ValueError("No debate to score.")

        transcript = "\n".join([
            f"{'User' if m['role'] == 'user' else 'AI'}: {m['content']}"
            for m in self.conversation_history
        ])

        # Build fallacy summary for the prompt
        if self.detected_fallacies:
            fallacy_summary = "Fallacies detected in user's arguments:\n" + "\n".join([
                f"- {f['name']}: {f['explanation']}"
                for f in self.detected_fallacies
            ])
        else:
            fallacy_summary = "No logical fallacies detected in user's arguments."

        scoring_prompt = f"""You are a strict, impartial debate judge. Score this debate honestly — do not favor either side.

    Topic: {self.topic}
    User's position: {self.user_position}
    AI's position: opposite of user

    {fallacy_summary}

    Debate transcript:
    {transcript}

    Score each side (0-10) on these 4 criteria:
    1. argument_strength — how strong and convincing were their core claims
    2. evidence_usage — did they back claims with real evidence
    3. logical_consistency — were their arguments logically sound (deduct for fallacies)
    4. handling_opponent — did they actually address the other side's points

    Important rules:
    - Be honest. If the user made strong points, reflect that in the score.
    - Deduct from user's logical_consistency if fallacies were detected.
    - Total = sum of all 4 scores.
    - Winner is whoever has the higher total. Use "draw" only if totals are equal.

    Return ONLY this JSON, no extra text:
    {{
    "user_score": {{
        "argument_strength": int,
        "evidence_usage": int,
        "logical_consistency": int,
        "handling_opponent": int,
        "total": int,
        "verdict": "2 sentences on how the user performed"
    }},
    "ai_score": {{
        "argument_strength": int,
        "evidence_usage": int,
        "logical_consistency": int,
        "handling_opponent": int,
        "total": int,
        "verdict": "2 sentences on how the AI performed"
    }},
    "winner": "user or ai or draw",
    "fallacies_used": {json.dumps(self.detected_fallacies)},
    "summary": "3 sentences summarizing the debate and why the winner won"
    }}
    """

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": scoring_prompt}],
                temperature=0.3,
                max_tokens=1000
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown code blocks if model wraps response in them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            scores = json.loads(raw.strip())
            return scores

        except json.JSONDecodeError:
            # If JSON parsing fails, return a safe fallback
            return {
                "user_score": {"argument_strength": 0, "evidence_usage": 0, "logical_consistency": 0, "handling_opponent": 0, "total": 0, "verdict": "Could not score."},
                "ai_score": {"argument_strength": 0, "evidence_usage": 0, "logical_consistency": 0, "handling_opponent": 0, "total": 0, "verdict": "Could not score."},
                "winner": "draw",
                "fallacies_used": self.detected_fallacies,
                "summary": "Scoring failed due to a parsing error."
            }