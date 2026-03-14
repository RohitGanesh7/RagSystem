"""
🏥 MedAgent – Hospital Multi-Agent System
Run:  streamlit run app.py
"""

import os
import streamlit as st
from agents import HospitalAgentSystem

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedAgent – Hospital AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
:root{
    --bg:#0d1117;--surface:#161b22;--surface2:#1f2937;
    --accent:#00d4aa;--accent2:#0ea5e9;
    --text:#e6edf3;--muted:#8b949e;--border:#30363d;
}
*{font-family:'DM Sans',sans-serif;}
.stApp{background:var(--bg);color:var(--text);}
#MainMenu,footer,.stDeployButton{display:none!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}

.hosp-header{text-align:center;padding:1.6rem 0 1rem;border-bottom:1px solid var(--border);margin-bottom:1.1rem;}
.hosp-header h1{font-family:'DM Serif Display',serif;font-size:2.1rem;color:var(--text);margin:0;}
.hosp-header h1 span{color:var(--accent);}
.hosp-header p{color:var(--muted);font-size:.83rem;margin:.3rem 0 0;}

.pipeline{display:flex;margin-bottom:1.1rem;background:var(--surface);border:1px solid var(--border);border-radius:10px;overflow:hidden;}
.step{flex:1;padding:.55rem .3rem;text-align:center;font-size:.7rem;font-weight:500;color:var(--muted);border-right:1px solid var(--border);transition:all .3s;}
.step:last-child{border-right:none;}
.step.active{background:rgba(0,212,170,.08);color:var(--accent);}
.step.done{background:rgba(0,212,170,.04);color:#6ee7b7;}
.step .icon{font-size:.95rem;display:block;}

.chat-wrap{max-height:480px;overflow-y:auto;padding:.3rem 0;margin-bottom:.6rem;}
.msg-row{display:flex;margin:.45rem 0;gap:.6rem;align-items:flex-start;}
.msg-row.user{flex-direction:row-reverse;}
.avatar{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.95rem;flex-shrink:0;}
.av-sys{background:linear-gradient(135deg,var(--accent),var(--accent2));}
.av-usr{background:var(--surface2);border:1px solid var(--border);}
.bubble{max-width:74%;padding:.75rem 1rem;border-radius:13px;font-size:.86rem;line-height:1.6;}
.bub-sys{background:var(--surface);border:1px solid var(--border);border-top-left-radius:4px;color:var(--text);}
.bub-usr{background:linear-gradient(135deg,#0e4f3e,#0d3a4a);border:1px solid rgba(0,212,170,.2);border-top-right-radius:4px;color:var(--text);}

.badge{display:inline-flex;align-items:center;gap:.3rem;padding:.16rem .55rem;border-radius:99px;font-size:.66rem;font-weight:600;letter-spacing:.4px;text-transform:uppercase;margin-bottom:.4rem;}
.b-triage{background:rgba(124,58,237,.15);color:#a78bfa;border:1px solid rgba(124,58,237,.3);}
.b-doctor{background:rgba(14,165,233,.15);color:#38bdf8;border:1px solid rgba(14,165,233,.3);}
.b-booking{background:rgba(16,185,129,.15);color:#34d399;border:1px solid rgba(16,185,129,.3);}
.b-ask{background:rgba(245,158,11,.15);color:#fbbf24;border:1px solid rgba(245,158,11,.3);}

/* Doctor selection section */
.doc-section-title{
    font-size:.82rem;font-weight:600;color:var(--muted);
    text-transform:uppercase;letter-spacing:.5px;
    margin:1rem 0 .5rem;padding-top:.8rem;
    border-top:1px solid var(--border);
}
.doc-grid{display:flex;flex-direction:column;gap:.5rem;margin-bottom:1rem;}
.doc-card{
    background:var(--surface);border:1px solid var(--border);border-radius:11px;
    padding:.85rem 1.1rem;display:flex;align-items:center;
    justify-content:space-between;transition:border-color .2s,background .2s;
}
.doc-card:hover{border-color:rgba(0,212,170,.5);background:rgba(0,212,170,.03);}
.doc-name{font-weight:600;font-size:.9rem;color:var(--text);}
.doc-meta{color:var(--muted);font-size:.76rem;margin-top:.15rem;}
.doc-time-badge{
    background:rgba(0,212,170,.1);border:1px solid rgba(0,212,170,.25);
    color:var(--accent);border-radius:7px;
    padding:.25rem .7rem;font-size:.8rem;font-weight:600;white-space:nowrap;
}

.stTextInput>div>div>input{background:var(--surface)!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:10px!important;padding:.68rem 1rem!important;font-size:.87rem!important;}
.stTextInput>div>div>input:focus{border-color:var(--accent)!important;box-shadow:0 0 0 2px rgba(0,212,170,.15)!important;}
.stButton>button{background:linear-gradient(135deg,var(--accent),var(--accent2))!important;color:#000!important;border:none!important;border-radius:10px!important;font-weight:600!important;transition:all .2s!important;}
.stButton>button:hover{opacity:.9;}

.info-card{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:.8rem 1rem;margin-bottom:.7rem;font-size:.81rem;}
.info-card h5{color:var(--accent);margin:0 0 .3rem;font-size:.73rem;text-transform:uppercase;letter-spacing:.4px;}
.info-card p{color:var(--muted);margin:0;line-height:1.5;}
hr{border-color:var(--border)!important;}
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ────────────────────────────────────────────────────────
defaults = {
    "messages":       [],
    "stage":          "greeting",   # greeting → confirming → confirmed
    "context":        {},
    "agent_system":   HospitalAgentSystem(),
    "pending_booking": None,        # stores chosen doctor dict until booking runs
    "input_key":       0,           # rotated to clear text_input widget
    "submitted_msg":   None,        # message to process THIS rerun only
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:.8rem 0;'>
        <div style='font-size:2.3rem'>🏥</div>
        <div style='font-family:DM Serif Display,serif;font-size:1.25rem;color:#e6edf3;'>MedAgent</div>
        <div style='color:#8b949e;font-size:.73rem;'>AI Hospital Assistant</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    optgpt_url   = st.text_input("🌐 OptGPT Server URL",
                                  value=os.environ.get("OPTGPT_API_URL","http://192.168.1.117:8006"))
    optgpt_model = st.text_input("🤖 Model", value="OptGPT-4:latest")

    if st.button("🔌 Apply", use_container_width=True):
        st.session_state.agent_system.set_api_url(optgpt_url, optgpt_model)
        st.rerun()

    # Always keep agent system in sync with sidebar values
    st.session_state.agent_system.set_api_url(optgpt_url, optgpt_model)

    crew_ok = st.session_state.agent_system.crew_llm is not None
    if crew_ok:
        st.success("✅ CrewAI + OptGPT ready")
    else:
        st.warning("⚡ Direct OptGPT mode\n(install litellm for full CrewAI)")

    st.markdown("---")
    st.markdown("""
    <div class='info-card'>
        <h5>🤖 Agent Pipeline</h5>
        <p><b style='color:#a78bfa'>①</b> Triage Agent<br>
           <b style='color:#38bdf8'>②</b> Doctor Agent<br>
           <b style='color:#34d399'>③</b> Booking Agent</p>
    </div>
    <div class='info-card'>
        <h5>💡 Try saying...</h5>
        <p>"I have fever and headache"<br>"Chest pain since morning"<br>"Stomach pain and nausea"<br>"Severe knee joint pain"</p>
    </div>""", unsafe_allow_html=True)

    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages        = []
        st.session_state.stage           = "greeting"
        st.session_state.context         = {}
        st.session_state.pending_booking = None
        st.session_state.submitted_msg   = None
        st.session_state.input_key       += 1
        st.rerun()

    st.markdown("---")
    st.markdown("<p style='color:#8b949e;font-size:.68rem;text-align:center;'>OptGPT · CrewAI · Streamlit</p>",
                unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hosp-header'>
    <h1>🏥 Med<span>Agent</span></h1>
    <p>AI-Powered Hospital Appointment System · Powered by OptGPT + CrewAI</p>
</div>""", unsafe_allow_html=True)

# ─── Pipeline Bar ─────────────────────────────────────────────────────────────
stages    = ["greeting", "confirming", "confirmed"]
stage_idx = stages.index(st.session_state.stage)
steps     = [("🩺","Symptoms & Triage"), ("📋","Choose Doctor"), ("✅","Confirmed")]
bar       = "<div class='pipeline'>"
for i,(icon,label) in enumerate(steps):
    css = "done" if i < stage_idx else ("active" if i == stage_idx else "")
    bar += f"<div class='step {css}'><span class='icon'>{icon}</span>{label}</div>"
bar += "</div>"
st.markdown(bar, unsafe_allow_html=True)

# ─── Auto greeting ────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "system", "type": "greeting",
        "content": (
            "Hello! 👋 I'm your AI hospital assistant.\n\n"
            "Please describe your **symptoms** and I'll connect you with the right doctor.\n\n"
            "*Example: \"I have fever and headache since yesterday\"*"
        )
    })

# ─── Render Chat Messages ─────────────────────────────────────────────────────
chat_html = "<div class='chat-wrap' id='chat-end'>"
for msg in st.session_state.messages:
    content = msg["content"].replace("\n","<br>")
    mtype   = msg.get("type","")

    badge = {
        "triage":  "<span class='badge b-triage'>⚡ Triage Agent</span><br>",
        "doctors": "<span class='badge b-doctor'>🩺 Doctor Agent</span><br>",
        "booking": "<span class='badge b-booking'>✅ Booking Agent</span><br>",
        "ask":     "<span class='badge b-ask'>❓ Select Doctor</span><br>",
    }.get(mtype, "")

    if msg["role"] == "user":
        chat_html += f"""
        <div class='msg-row user'>
            <div class='avatar av-usr'>👤</div>
            <div class='bubble bub-usr'>{content}</div>
        </div>"""
    else:
        chat_html += f"""
        <div class='msg-row'>
            <div class='avatar av-sys'>🏥</div>
            <div class='bubble bub-sys'>{badge}{content}</div>
        </div>"""
chat_html += "</div>"
st.markdown(chat_html, unsafe_allow_html=True)


# ─── STEP 1: Process pending booking (a button was clicked last rerun) ────────
# We check FIRST before rendering anything interactive, so the booking
# happens at the start of the next rerun — not mid-render.
if st.session_state.pending_booking is not None and st.session_state.stage == "confirming":
    chosen = st.session_state.pending_booking
    st.session_state.pending_booking = None   # clear immediately so it doesn't re-fire

    choice_text = f"Book {chosen['name']} at {chosen['time']}"
    st.session_state.messages.append({"role": "user", "content": choice_text})

    with st.spinner(f"📅 Booking Agent confirming with {chosen['name']} via OptGPT..."):
        booking = st.session_state.agent_system.run_booking(
            choice_text,
            st.session_state.context.get("doctors", {}),
            st.session_state.context.get("triage", {}),
        )

    st.session_state.messages.append({
        "role": "system", "type": "booking", "content": booking["display"]
    })
    st.session_state.stage = "confirmed"
    st.rerun()


# ─── STEP 2: Doctor selection cards (only shown in 'confirming' stage) ────────
if st.session_state.stage == "confirming":
    doctors  = st.session_state.context.get("doctors", {}).get("doctors", [])
    date     = st.session_state.context.get("doctors", {}).get("date", "Tomorrow")
    triage   = st.session_state.context.get("triage", {})
    spec     = triage.get("specialist", "Doctor")

    st.markdown(
        f"<div class='doc-section-title'>👇 Select your {spec} — {date}</div>",
        unsafe_allow_html=True
    )

    for doc in doctors:
        col_info, col_btn = st.columns([5, 1])
        with col_info:
            st.markdown(f"""
            <div class='doc-card'>
                <div>
                    <div class='doc-name'>👨‍⚕️ {doc['name']}</div>
                    <div class='doc-meta'>🚪 Room {doc['room']}</div>
                </div>
                <div class='doc-time-badge'>🕐 {doc['time']}</div>
            </div>""", unsafe_allow_html=True)
        with col_btn:
            st.write("")   # vertical alignment spacer
            if st.button("Book →", key=f"book_{doc['number']}_{doc['name']}"):
                # Store the choice — DO NOT run booking here.
                # Streamlit will rerun and we handle it at the top (STEP 1).
                st.session_state.pending_booking = doc
                st.rerun()

    st.markdown(
        "<p style='color:var(--muted);font-size:.76rem;margin:.3rem 0 .8rem;'>"
        "Or type your preference below ↓</p>",
        unsafe_allow_html=True
    )


# ─── STEP 3: Input bar ────────────────────────────────────────────────────────
if st.session_state.stage == "confirmed":
    st.success("🎉 Appointment booked! Click **New Conversation** in the sidebar to start again.")
else:
    st.markdown("---")
    placeholder = {
        "greeting":   "Describe your symptoms...",
        "confirming": "Or type: 'Book Dr. Ramesh at 10:30 AM'",
    }.get(st.session_state.stage, "")

    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "msg", placeholder=placeholder,
            label_visibility="collapsed",
            key=f"user_msg_{st.session_state.input_key}"   # key rotation clears the field
        )
    with col2:
        send = st.button("Send →", use_container_width=True)

    # ── Capture submission — only when Send clicked OR Enter pressed ──────────
    # We store the message into submitted_msg and clear the input widget by
    # bumping input_key. On the NEXT rerun submitted_msg holds the message.
    if send and user_input.strip():
        st.session_state.submitted_msg = user_input.strip()
        st.session_state.input_key    += 1    # clears the text box on next render
        st.rerun()

    # ── Process the stored submission ─────────────────────────────────────────
    if st.session_state.submitted_msg:
        msg = st.session_state.submitted_msg
        st.session_state.submitted_msg = None  # consume it immediately

        st.session_state.messages.append({"role": "user", "content": msg})

        # ── Symptom intake → run triage + doctor finder ───────────────────────
        if st.session_state.stage == "greeting":
            with st.spinner("🔬 Triage Agent analyzing symptoms via OptGPT..."):
                triage = st.session_state.agent_system.run_triage(msg)
            st.session_state.context["triage"] = triage
            st.session_state.messages.append(
                {"role":"system","type":"triage","content":triage["display"]}
            )

            with st.spinner("📋 Doctor Agent finding availability via OptGPT..."):
                doctors = st.session_state.agent_system.run_doctor_finder(triage["specialist"])
            st.session_state.context["doctors"] = doctors
            st.session_state.messages.append(
                {"role":"system","type":"doctors","content":doctors["display"]}
            )

            st.session_state.messages.append({
                "role":"system","type":"ask",
                "content":(
                    f"Based on your symptoms, a **{triage['specialist']}** is recommended.\n\n"
                    f"👇 **Please select a doctor using the cards below**, or type your choice."
                )
            })
            st.session_state.stage = "confirming"
            st.rerun()

        # ── Patient typed their choice instead of clicking a card ─────────────
        elif st.session_state.stage == "confirming":
            with st.spinner("📅 Booking Agent confirming appointment via OptGPT..."):
                booking = st.session_state.agent_system.run_booking(
                    msg,
                    st.session_state.context.get("doctors", {}),
                    st.session_state.context.get("triage", {}),
                )
            st.session_state.messages.append(
                {"role":"system","type":"booking","content":booking["display"]}
            )
            st.session_state.stage = "confirmed"
            st.rerun()

st.markdown(
    "<script>var c=document.getElementById('chat-end');if(c)c.scrollTop=c.scrollHeight;</script>",
    unsafe_allow_html=True
)
