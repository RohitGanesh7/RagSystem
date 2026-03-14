"""
🏥 Hospital Multi-Agent System – Agents Module
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LLM Strategy:
  ① CrewAI + LiteLLM  →  openai/<model> with custom base_url → OptGPT /v1/chat/completions
  ② Direct httpx      →  OptGPT /api/generate  (streaming, your original code)
  ③ Keyword fallback  →  pure Python rules, no network

The direct httpx path uses YOUR EXACT generate_text() logic — untouched.
"""

import os, json, asyncio, random
from datetime import datetime, timedelta
from typing import Optional

import httpx

# ── CrewAI ────────────────────────────────────────────────────────────────────
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.llm import LLM as CrewLLM
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
OPTGPT_API_URL_ENV_KEYS = ["OPTGPT_API_URL", "REACT_APP_OPTGPT_API_URL"]
DEFAULT_OPTGPT_URL      = "http://192.168.1.117:8006"
DEFAULT_MODEL_NAME      = "OptGPT-4:latest"

VALID_SPECIALISTS = [
    "General Physician", "Cardiologist", "Dermatologist",
    "Gastroenterologist", "Neurologist", "Orthopedic",
    "Pulmonologist", "ENT Specialist",
]

# ── Mock Hospital DB ──────────────────────────────────────────────────────────
DOCTORS_DB = {
    "General Physician": [
        {"name": "Dr. Ramesh Kumar",  "room": "101", "slots": ["9:00 AM", "10:30 AM", "2:00 PM", "4:30 PM"]},
        {"name": "Dr. Kavita Sharma", "room": "102", "slots": ["10:00 AM", "12:00 PM", "3:00 PM"]},
        {"name": "Dr. Ahmed Hassan",  "room": "103", "slots": ["9:30 AM", "11:00 AM", "4:00 PM", "5:00 PM"]},
    ],
    "Cardiologist": [
        {"name": "Dr. Priya Mehta",   "room": "201", "slots": ["9:00 AM", "11:30 AM", "3:30 PM"]},
        {"name": "Dr. Suresh Patel",  "room": "202", "slots": ["10:00 AM", "1:00 PM", "4:00 PM"]},
    ],
    "Dermatologist": [
        {"name": "Dr. Anjali Singh",  "room": "301", "slots": ["9:30 AM", "11:00 AM", "2:30 PM"]},
        {"name": "Dr. Raj Gupta",     "room": "302", "slots": ["10:30 AM", "12:00 PM", "4:00 PM"]},
    ],
    "Gastroenterologist": [
        {"name": "Dr. Meena Iyer",    "room": "401", "slots": ["9:00 AM", "11:00 AM", "2:00 PM"]},
        {"name": "Dr. Vikram Nair",   "room": "402", "slots": ["10:00 AM", "1:30 PM", "4:30 PM"]},
    ],
    "Neurologist": [
        {"name": "Dr. Arun Krishnan", "room": "501", "slots": ["9:00 AM", "11:30 AM", "3:00 PM"]},
        {"name": "Dr. Shalini Das",   "room": "502", "slots": ["10:00 AM", "1:00 PM", "4:00 PM"]},
    ],
    "Orthopedic": [
        {"name": "Dr. Sanjay Reddy",  "room": "601", "slots": ["9:30 AM", "12:00 PM", "2:30 PM", "5:00 PM"]},
    ],
    "Pulmonologist": [
        {"name": "Dr. Neha Joshi",    "room": "701", "slots": ["10:00 AM", "1:00 PM", "3:30 PM"]},
    ],
    "ENT Specialist": [
        {"name": "Dr. Kiran Bose",    "room": "801", "slots": ["9:00 AM", "11:00 AM", "2:00 PM"]},
    ],
}

SYMPTOM_MAP = {
    "fever":     "General Physician",  "headache":  "General Physician",
    "cold":      "General Physician",  "flu":       "General Physician",
    "fatigue":   "General Physician",  "weakness":  "General Physician",
    "cough":     "Pulmonologist",      "breathing": "Pulmonologist",
    "breath":    "Pulmonologist",      "wheezing":  "Pulmonologist",
    "chest":     "Cardiologist",       "heart":     "Cardiologist",
    "palpitat":  "Cardiologist",
    "skin":      "Dermatologist",      "rash":      "Dermatologist",
    "itch":      "Dermatologist",      "acne":      "Dermatologist",
    "stomach":   "Gastroenterologist", "nausea":    "Gastroenterologist",
    "vomit":     "Gastroenterologist", "diarrhea":  "Gastroenterologist",
    "abdomen":   "Gastroenterologist", "bloat":     "Gastroenterologist",
    "migraine":  "Neurologist",        "seizure":   "Neurologist",
    "numbness":  "Neurologist",        "dizziness": "Neurologist",
    "bone":      "Orthopedic",         "joint":     "Orthopedic",
    "knee":      "Orthopedic",         "fracture":  "Orthopedic",
    "ear":       "ENT Specialist",     "throat":    "ENT Specialist",
    "sinus":     "ENT Specialist",     "nose":      "ENT Specialist",
}


# ══════════════════════════════════════════════════════════════════════════════
# YOUR ORIGINAL generate_text() — kept exactly as provided, zero changes
# Only renamed to _optgpt_generate for internal use, same signature/logic
# ══════════════════════════════════════════════════════════════════════════════

async def _optgpt_generate(
    prompt: str,
    base_url: str,                        # injected — not from env
    model: str = DEFAULT_MODEL_NAME,
    temperature: float = 0.7,
    max_tokens: int = 10000,
    top_p: float = 0.95,
    n_predict: int = 6000,
    timeout: Optional[float] = None,
) -> str:
    """
    Call the OptGPT streaming API and return the full generated text.
    Mirrors the behavior of the provided Node.js axios implementation.
    """
    url = f"{base_url}/api/generate"

    payload = {
        "model":       model,
        "prompt":      prompt,
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "top_p":       top_p,
        "n_predict":   n_predict,
    }

    full_response = ""
    buffer        = ""

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()

            async for chunk in resp.aiter_text():
                buffer += chunk
                lines   = buffer.split("\n")
                buffer  = lines.pop() or ""

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, dict) and parsed.get("response"):
                            full_response += str(parsed["response"])
                    except json.JSONDecodeError:
                        # Skip invalid JSON lines, same as the Node code
                        continue

    if not full_response.strip():
        raise RuntimeError("No content generated from OptGPT.")

    return full_response.strip()


# ── Sync runner for the async client ─────────────────────────────────────────
def _run_sync(coro):
    import concurrent.futures
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _extract_json(text: str) -> dict:
    """Pull first JSON object from LLM output (handles markdown fences)."""
    if "```" in text:
        for part in text.split("```"):
            part = part.strip().lstrip("json").strip()
            if part.startswith("{"):
                try:
                    return json.loads(part)
                except Exception:
                    pass
    s, e = text.find("{"), text.rfind("}") + 1
    if s != -1 and e > s:
        try:
            return json.loads(text[s:e])
        except Exception:
            pass
    return {}


def _keyword_triage(text: str):
    tl = text.lower()
    symptoms, specialist = [], "General Physician"
    # multi-word patterns first
    for phrase, spec in [
        ("back pain",    "Orthopedic"),
        ("chest pain",   "Cardiologist"),
        ("body ache",    "General Physician"),
        ("shortness of breath", "Pulmonologist"),
    ]:
        if phrase in tl:
            symptoms.append(phrase.title())
            specialist = spec
    # single-keyword scan
    for kw, spec in SYMPTOM_MAP.items():
        if kw in tl:
            sym = kw.title()
            if sym not in symptoms:
                symptoms.append(sym)
            specialist = spec

    severity = (
        "severe"   if any(w in tl for w in ["severe","intense","unbearable","emergency","very bad"]) else
        "mild"     if any(w in tl for w in ["mild","slight","little","minor"]) else
        "moderate"
    )
    return symptoms or ["General Discomfort"], specialist, severity


# ══════════════════════════════════════════════════════════════════════════════
# HospitalAgentSystem
# ══════════════════════════════════════════════════════════════════════════════

class HospitalAgentSystem:
    """
    3 CrewAI agents backed by your OptGPT model.

    LLM call chain (each agent tries in order):
      A) CrewAI + LiteLLM  →  openai/<model> → OptGPT /v1/chat/completions
      B) Direct httpx       →  OptGPT /api/generate  (your original code)
      C) Keyword rules      →  no network needed
    """

    def __init__(self):
        self.base_url: str = DEFAULT_OPTGPT_URL
        self.model: str    = DEFAULT_MODEL_NAME
        self.crew_llm      = None
        self.api_configured = False

        for key in OPTGPT_API_URL_ENV_KEYS:
            val = os.getenv(key)
            if val:
                self.base_url = val.rstrip("/")
                break

        self._build_crew_llm()

    def set_api_url(self, url: str, model: str = DEFAULT_MODEL_NAME):
        self.base_url = url.rstrip("/")
        self.model    = model
        self._build_crew_llm()

    def _build_crew_llm(self):
        """
        Build CrewAI LLM using LiteLLM so we can use any custom OpenAI-compat
        endpoint.  Requires:  pip install litellm
        OptGPT must expose  /v1/chat/completions  (OpenAI-compatible).
        """
        self.api_configured = True
        self.crew_llm = None

        if not CREWAI_AVAILABLE:
            print("[MedAgent] CrewAI not installed — using direct OptGPT only")
            return

        try:
            import litellm  # noqa: F401  (just check it's installed)
            self.crew_llm = CrewLLM(
                model     = f"openai/{self.model}",
                base_url  = f"{self.base_url}/v1",  # OptGPT OpenAI-compat endpoint
                api_key   = "optgpt-local",          # dummy — OptGPT doesn't validate
                temperature = 0.3,
                max_tokens  = 800,
            )
            print(f"[MedAgent] ✅ CrewAI+LiteLLM ready → {self.base_url}/v1  model={self.model}")
        except ImportError:
            print("[MedAgent] litellm not installed — pip install litellm")
            print("[MedAgent] Falling back to direct OptGPT calls")
        except Exception as e:
            print(f"[MedAgent] CrewLLM init failed: {e}")

    # ── Path A: CrewAI + LiteLLM ─────────────────────────────────────────────
    def _run_crew(self, role: str, goal: str, backstory: str, task_desc: str) -> Optional[str]:
        if not CREWAI_AVAILABLE or self.crew_llm is None:
            return None
        try:
            agent = Agent(
                role=role, goal=goal, backstory=backstory,
                llm=self.crew_llm,
                verbose=False, allow_delegation=False, max_iter=2,
            )
            task = Task(description=task_desc, agent=agent, expected_output="Valid JSON object")
            crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
            return str(crew.kickoff()).strip()
        except Exception as e:
            print(f"[CrewAI] {e}")
            return None

    # ── Path B: direct OptGPT /api/generate (your original code) ─────────────
    def _run_direct(self, prompt: str) -> Optional[str]:
        try:
            result = _run_sync(_optgpt_generate(
                prompt     = prompt,
                base_url   = self.base_url,
                model      = self.model,
                temperature = 0.3,
                max_tokens  = 800,
                n_predict   = 800,
                timeout     = 60.0,
            ))
            print("[MedAgent] ✅ Direct OptGPT responded")
            return result
        except Exception as e:
            print(f"[OptGPT direct] {e}")
            return None

    # ── Combined: A → B → None (C handled per-agent) ─────────────────────────
    def _llm(self, role: str, goal: str, backstory: str, task_desc: str) -> Optional[str]:
        raw = self._run_crew(role, goal, backstory, task_desc)
        if raw:
            print("[MedAgent] ✅ CrewAI+OptGPT responded")
            return raw
        print("[MedAgent] ⚠️  CrewAI failed, trying direct OptGPT...")
        return self._run_direct(
            f"You are {role}. {goal}\n\n{backstory}\n\nTask:\n{task_desc}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # AGENT 1 – Triage
    # ══════════════════════════════════════════════════════════════════════════

    def run_triage(self, patient_message: str) -> dict:
        task_desc = f"""A patient says: "{patient_message}"

Analyze the symptoms. Reply ONLY with a valid JSON object — no markdown, no explanation:
{{
  "symptoms": ["Symptom1", "Symptom2"],
  "severity": "moderate",
  "specialist": "General Physician",
  "reasoning": "one line"
}}
severity must be exactly: mild | moderate | severe
specialist must be exactly one of: {', '.join(VALID_SPECIALISTS)}"""

        raw = self._llm(
            role      = "Medical Triage Specialist",
            goal      = "Map patient symptoms to the correct medical specialist",
            backstory = "Senior triage nurse, 15 years experience. Always responds with compact valid JSON only.",
            task_desc = task_desc,
        )

        if raw:
            data = _extract_json(raw)
            if data.get("specialist") and data.get("symptoms"):
                if data["specialist"] not in VALID_SPECIALISTS:
                    data["specialist"] = "General Physician"
                return self._fmt_triage(data["symptoms"], data["specialist"], data.get("severity","moderate"))

        # Path C – keyword rules
        syms, spec, sev = _keyword_triage(patient_message)
        return self._fmt_triage(syms, spec, sev)

    def _fmt_triage(self, symptoms, specialist, severity) -> dict:
        icon = {"mild":"🟡","moderate":"🟠","severe":"🔴"}.get(severity,"🟠")
        display = (
            f"**Symptoms Detected:** {icon} *{severity.title()} severity*\n\n"
            + "\n".join(f"  • {s}" for s in symptoms)
            + f"\n\n**Recommended Specialist:** 🩺 {specialist}"
        )
        return {"display":display, "symptoms":symptoms, "specialist":specialist, "severity":severity}

    # ══════════════════════════════════════════════════════════════════════════
    # AGENT 2 – Doctor Finder
    # ══════════════════════════════════════════════════════════════════════════

    def run_doctor_finder(self, specialist: str) -> dict:
        db       = DOCTORS_DB.get(specialist, DOCTORS_DB["General Physician"])
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%A, %B %d")
        db_str   = "; ".join(
            f"{d['name']} room {d['room']} slots [{', '.join(d['slots'][:3])}]"
            for d in db[:3]
        )

        task_desc = f"""List 3 available {specialist} doctors for TOMORROW ({tomorrow}).
Use ONLY names/rooms from: {db_str}
Pick one different time slot per doctor.

Reply ONLY with valid JSON — no markdown:
{{
  "doctors": [
    {{"number":1, "name":"Dr. Full Name", "time":"10:30 AM", "room":"101"}},
    {{"number":2, "name":"Dr. Full Name", "time":"2:00 PM",  "room":"102"}},
    {{"number":3, "name":"Dr. Full Name", "time":"4:30 PM",  "room":"103"}}
  ]
}}"""

        raw = self._llm(
            role      = "Hospital Scheduling Coordinator",
            goal      = "Find available doctors and time slots for a specialty",
            backstory = "Hospital scheduler. Returns exact names/rooms from provided data in compact JSON only.",
            task_desc = task_desc,
        )

        if raw:
            data = _extract_json(raw)
            if data.get("doctors") and len(data["doctors"]) >= 1:
                return self._fmt_doctors(data["doctors"], specialist, tomorrow)

        # Fallback: build from DB directly
        doctors = [
            {"number":i+1,"name":d["name"],"time":random.choice(d["slots"]),"room":d["room"]}
            for i,d in enumerate(db[:3])
        ]
        return self._fmt_doctors(doctors, specialist, tomorrow)

    def _fmt_doctors(self, doctors, specialist, date) -> dict:
        lines = [f"**Available {specialist} Doctors – {date}:**\n"]
        for d in doctors:
            lines.append(f"  **{d['number']}.** {d['name']} – {d['time']}  *(Room {d['room']})*")
        return {"display":"\n".join(lines), "doctors":doctors, "specialist":specialist, "date":date}

    # ══════════════════════════════════════════════════════════════════════════
    # AGENT 3 – Booking
    # ══════════════════════════════════════════════════════════════════════════

    def run_booking(self, patient_choice: str, doctor_context: dict, triage_context: dict) -> dict:
        date       = doctor_context.get("date","Tomorrow")
        specialist = doctor_context.get("specialist","General Physician")
        docs_json  = json.dumps(doctor_context.get("doctors",[]))

        task_desc = f"""Patient said: "{patient_choice}"
Available options: {docs_json}
Date: {date} | Specialty: {specialist}

Match patient's choice. Generate confirmation ID like HOSP-48291.
Reply ONLY with valid JSON — no markdown:
{{
  "doctor":          "Dr. Full Name",
  "time":            "10:30 AM",
  "date":            "{date}",
  "room":            "101",
  "confirmation_id": "HOSP-XXXXX",
  "specialist":      "{specialist}"
}}"""

        raw = self._llm(
            role      = "Appointment Booking Specialist",
            goal      = "Book the correct appointment and return a confirmed booking in JSON",
            backstory = "Hospital booking agent. Matches patient requests to available slots. Compact JSON only.",
            task_desc = task_desc,
        )

        if raw:
            data = _extract_json(raw)
            if data.get("doctor") and data.get("time"):
                data.setdefault("confirmation_id", f"HOSP-{random.randint(10000,99999)}")
                data.setdefault("date", date)
                data.setdefault("specialist", specialist)
                return self._fmt_booking(data)

        return self._fallback_booking(patient_choice, doctor_context)

    def _fallback_booking(self, choice, ctx) -> dict:
        doctors = ctx.get("doctors", [])
        cl      = choice.lower()
        matched = next(
            (d for d in doctors if any(p in cl for p in d["name"].lower().split() if len(p) > 2)),
            doctors[0] if doctors else {"name":"Dr. Ramesh Kumar","time":"10:30 AM","room":"101"},
        )
        return self._fmt_booking({
            "doctor":          matched["name"],
            "time":            matched["time"],
            "date":            ctx.get("date","Tomorrow"),
            "room":            matched.get("room","101"),
            "confirmation_id": f"HOSP-{random.randint(10000,99999)}",
            "specialist":      ctx.get("specialist","General Physician"),
        })

    def _fmt_booking(self, data) -> dict:
        display = (
            f"✅ **Appointment Confirmed!**\n\n---\n"
            f"📋 **Confirmation ID:** `{data.get('confirmation_id')}`\n\n"
            f"👨‍⚕️ **Doctor:** {data.get('doctor')}\n"
            f"🏥 **Specialty:** {data.get('specialist')}\n"
            f"📅 **Date:** {data.get('date')}\n"
            f"🕐 **Time:** {data.get('time')}\n"
            f"🚪 **Room:** {data.get('room')}\n\n---\n"
            f"*Please arrive 10 minutes early. Bring valid ID and previous prescriptions.*\n"
            f"*Cancellations: 📞 +91-9876-543210*"
        )
        return {"display":display, "details":data}
