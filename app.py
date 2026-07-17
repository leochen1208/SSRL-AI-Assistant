import streamlit as st

from auth import check_login
from google_sheet import (
    get_or_create_room_state,
    get_room_messages,
    save_chat,
    save_room_message,
    start_new_room_discussion,
    update_room_expected_phase,
)
from openai_client import (
    analyze_quality_state,
    analyze_ssrl_state,
    ask_ai,
)

STUDENT_AVATARS = ["🐱", "🐻", "🦝", "🐹", "🐮", "🦄"]
AI_AVATAR = "🤖"

ALLOWED_CONDITIONS = {"structure", "quality", "hybrid"}

PHASE_ORDER = {
    "general": 0,
    "content_discussion": 0,
    "social_response": 0,
    "unclear": 0,
    "task_understanding": 1,
    "planning": 2,
    "monitoring": 3,
    "reflection": 4,
}

VALID_PHASES = set(PHASE_ORDER)
VALID_QUALITY_TRIGGER_TYPES = {
    "negative_value_monitoring",
    "concept_confusion",
    "lack_of_shared_perspective",
    "none",
}

MIN_MESSAGES_BEFORE_STRUCTURE_INTERVENTION = 2
MIN_PHASE_CONFIDENCE = 0.75


def get_student_avatar(sender_id: str) -> str:
    sender_id = str(sender_id).strip()
    if not sender_id:
        return "⚪"
    avatar_index = sum(ord(character) for character in sender_id) % len(STUDENT_AVATARS)
    return STUDENT_AVATARS[avatar_index]


st.set_page_config(
    page_title="SSRL AI Assistant",
    page_icon="🤖",
    layout="centered",
)


def initialize_session_state() -> None:
    default_values = {
        "login": False,
        "username": "",
        "name": "",
        "condition": "",
        "room_id": "",
        "session_id": "",
        "expected_phase": "task_understanding",
        "room_state_initialized": False,
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


def clear_login_state() -> None:
    st.session_state.login = False
    st.session_state.username = ""
    st.session_state.name = ""
    st.session_state.condition = ""
    st.session_state.room_id = ""
    st.session_state.session_id = ""
    st.session_state.expected_phase = "task_understanding"
    st.session_state.room_state_initialized = False


def logout() -> None:
    clear_login_state()


def apply_room_state(room_state: dict) -> None:
    st.session_state.session_id = str(room_state.get("session_id", "")).strip()
    st.session_state.expected_phase = str(
        room_state.get("expected_phase", "task_understanding")
    ).strip() or "task_understanding"
    st.session_state.room_state_initialized = True


def refresh_room_state() -> dict:
    room_state = get_or_create_room_state(
        room_id=st.session_state.room_id,
        condition=st.session_state.condition,
    )
    apply_room_state(room_state)
    return room_state


def show_login_page() -> None:
    st.title("🤖 SSRL AI Assistant")
    st.write("請輸入帳號與密碼進入討論系統。")

    username = st.text_input("帳號", key="login_username")
    password = st.text_input("密碼", type="password", key="login_password")

    if not st.button("登入", type="primary", use_container_width=True):
        return

    username = username.strip()
    password = password.strip()

    if not username or not password:
        st.warning("請輸入帳號與密碼。")
        return

    try:
        user = check_login(username, password)
    except RuntimeError as error:
        st.error(str(error))
        return
    except Exception as error:
        st.error(f"登入系統發生未預期錯誤：{error}")
        return

    if user is None:
        st.error("帳號或密碼錯誤，或帳號已停用。")
        return

    condition = str(user.get("condition", "")).strip().lower()
    room_id = str(user.get("room_id", "")).strip()

    if condition not in ALLOWED_CONDITIONS:
        st.error("此帳號的實驗條件設定錯誤，請聯絡研究者。")
        return

    if not room_id:
        st.error("此帳號尚未設定聊天室編號，請聯絡研究者。")
        return

    st.session_state.login = True
    st.session_state.username = user["username"]
    st.session_state.name = user["name"]
    st.session_state.condition = condition
    st.session_state.room_id = room_id

    try:
        refresh_room_state()
    except Exception as error:
        clear_login_state()
        st.error(f"聊天室初始化失敗：{error}")
        return

    st.success("登入成功")
    st.rerun()


def load_room_messages() -> list[dict]:
    return get_room_messages(
        room_id=st.session_state.room_id,
        session_id=st.session_state.session_id,
    )


def build_student_history(room_messages: list[dict]) -> list[dict]:
    return [
        {
            "role": "user",
            "speaker": message.get("sender_name", "小組成員"),
            "content": message["content"],
        }
        for message in room_messages
        if message.get("role") == "user"
        and str(message.get("content", "")).strip()
    ]


def analyze_current_phase(
    student_history: list[dict],
    expected_phase_before: str,
) -> dict:
    try:
        result = analyze_ssrl_state(
            student_history=student_history,
            expected_phase=expected_phase_before,
        )
        if not isinstance(result, dict):
            raise TypeError("SSRL 階段分析結果不是字典格式。")
        return result
    except Exception as error:
        st.warning("階段分析發生問題，本次暫時記錄為 general。")
        return {
            "observed_phase": "general",
            "phase_completed": False,
            "next_expected_phase": expected_phase_before,
            "confidence": 0.0,
            "shared_evidence": False,
            "evidence_insufficient": True,
            "reason": f"階段分析失敗：{error}",
        }


def normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def normalize_confidence(value) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(confidence, 1.0))


def validate_phase_result(
    analysis_result: dict,
    expected_phase_before: str,
) -> tuple[str, bool, str, str, float, bool, bool]:
    observed_phase = str(
        analysis_result.get("observed_phase", "general")
    ).strip()
    phase_completed = normalize_bool(
        analysis_result.get("phase_completed", False)
    )
    next_expected_phase = str(
        analysis_result.get("next_expected_phase", expected_phase_before)
    ).strip()
    reason = str(analysis_result.get("reason", "")).strip()
    confidence = normalize_confidence(analysis_result.get("confidence", 0.0))
    shared_evidence = normalize_bool(
        analysis_result.get("shared_evidence", False)
    )
    evidence_insufficient = normalize_bool(
        analysis_result.get("evidence_insufficient", False)
    )

    if observed_phase not in VALID_PHASES:
        observed_phase = "unclear"

    if next_expected_phase not in VALID_PHASES:
        next_expected_phase = expected_phase_before

    if PHASE_ORDER.get(next_expected_phase, 0) == 0:
        next_expected_phase = expected_phase_before

    if confidence < MIN_PHASE_CONFIDENCE:
        phase_completed = False
    if not shared_evidence:
        phase_completed = False
    if evidence_insufficient:
        phase_completed = False

    return (
        observed_phase,
        phase_completed,
        next_expected_phase,
        reason,
        confidence,
        shared_evidence,
        evidence_insufficient,
    )


def analyze_current_quality(student_history: list[dict]) -> dict:
    try:
        result = analyze_quality_state(student_history=student_history)
        if not isinstance(result, dict):
            raise TypeError("品質分析結果不是字典格式。")
        return result
    except Exception as error:
        st.warning("品質分析發生問題，本次不觸發品質介入。")
        return {
            "triggered": False,
            "trigger_type": "none",
            "intervention": "",
            "reason": f"品質分析失敗：{error}",
        }


def validate_quality_result(
    quality_result: dict,
) -> tuple[bool, str, str, str]:
    triggered = normalize_bool(quality_result.get("triggered", False))
    trigger_type = str(quality_result.get("trigger_type", "none")).strip()
    intervention = str(quality_result.get("intervention", "")).strip()
    reason = str(quality_result.get("reason", "")).strip()

    if trigger_type not in VALID_QUALITY_TRIGGER_TYPES:
        trigger_type = "none"

    if not triggered or trigger_type == "none":
        triggered = False
        trigger_type = "none"
        intervention = ""

    return triggered, trigger_type, intervention, reason


def determine_structure_intervention(
    observed_phase: str,
    expected_phase_before: str,
    student_history: list[dict],
    confidence: float,
    evidence_insufficient: bool,
) -> bool:
    if observed_phase in {
        "general",
        "content_discussion",
        "social_response",
        "unclear",
    }:
        return False

    if len(student_history) < MIN_MESSAGES_BEFORE_STRUCTURE_INTERVENTION:
        return False

    if confidence < MIN_PHASE_CONFIDENCE or evidence_insufficient:
        return False

    observed_order = PHASE_ORDER.get(observed_phase, 0)
    expected_order = PHASE_ORDER.get(expected_phase_before, 1)
    return observed_order > expected_order


def generate_structure_intervention(
    student_history: list[dict],
    expected_phase: str,
    observed_phase: str,
) -> str:
    try:
        intervention_message = ask_ai(
            student_history=student_history,
            expected_phase=expected_phase,
            observed_phase=observed_phase,
        )
        return str(intervention_message or "").strip()
    except Exception as error:
        st.error(f"結構介入訊息產生失敗：{error}")
        return ""


def update_expected_phase(
    expected_phase_before: str,
    observed_phase: str,
    phase_completed: bool,
    next_expected_phase: str,
) -> str:
    if phase_completed and observed_phase == expected_phase_before:
        expected_phase_after = next_expected_phase
    else:
        expected_phase_after = expected_phase_before

    if expected_phase_after != expected_phase_before:
        update_room_expected_phase(
            room_id=st.session_state.room_id,
            expected_phase=expected_phase_after,
        )

    st.session_state.expected_phase = expected_phase_after
    return expected_phase_after


def decide_intervention(
    condition: str,
    structure_triggered: bool,
    quality_triggered: bool,
    quality_trigger_type: str,
    quality_intervention: str,
    expected_phase_before: str,
    observed_phase: str,
    student_history: list[dict],
) -> tuple[bool, str, str, str]:
    if condition in {"structure", "hybrid"} and structure_triggered:
        intervention = generate_structure_intervention(
            student_history=student_history,
            expected_phase=expected_phase_before,
            observed_phase=observed_phase,
        )
        return bool(intervention), "structure", "phase_skip", intervention

    if condition in {"quality", "hybrid"} and quality_triggered:
        return (
            bool(quality_intervention),
            "quality",
            quality_trigger_type,
            quality_intervention,
        )

    return False, "none", "none", ""


def save_student_record(
    user_input: str,
    observed_phase: str,
    expected_phase_before: str,
    expected_phase_after: str,
    phase_completed: bool,
    quality_triggered: bool,
    quality_trigger_type: str,
    should_intervene: bool,
    trigger_source: str,
    trigger_type: str,
    phase_reason: str,
    quality_reason: str,
    intervention_message: str,
) -> None:
    try:
        save_chat(
            student_id=st.session_state.username,
            session_id=st.session_state.session_id,
            condition=st.session_state.condition,
            role="user",
            message=user_input,
            observed_phase=observed_phase,
            expected_phase_before=expected_phase_before,
            expected_phase_after=expected_phase_after,
            phase_completed=phase_completed,
            quality_triggered=quality_triggered,
            quality_trigger_type=quality_trigger_type,
            should_intervene=should_intervene,
            trigger_source=trigger_source,
            trigger_type=trigger_type,
            phase_reason=phase_reason,
            quality_reason=quality_reason,
            intervention=intervention_message,
        )
    except Exception as error:
        st.warning(f"Google Sheet 分析紀錄失敗：{error}")


def render_message(message: dict) -> None:
    role = str(message.get("role", "user")).strip()
    sender_name = str(message.get("sender_name", "小組成員")).strip() or "小組成員"
    sender_id = str(message.get("sender_id", "")).strip()
    content = str(message.get("content", "")).strip()

    if not content:
        return

    if role == "assistant":
        with st.chat_message("assistant", avatar=AI_AVATAR):
            st.markdown("**AI 助理**")
            st.write(content)
        return

    with st.chat_message(
        sender_name,
        avatar=get_student_avatar(sender_id),
    ):
        st.markdown(f"**{sender_name}**")
        st.write(content)


@st.fragment(run_every="10s")
def render_shared_chat() -> None:
    try:
        room_messages = load_room_messages()
    except Exception as error:
        st.warning(f"聊天室同步失敗：{error}")
        return

    if not room_messages:
        st.info("目前尚無討論訊息。")
        return

    for message in room_messages:
        render_message(message)


def begin_new_discussion() -> None:
    room_state = start_new_room_discussion(
        room_id=st.session_state.room_id,
        condition=st.session_state.condition,
    )
    apply_room_state(room_state)


def process_user_message(user_input: str) -> None:
    save_room_message(
        room_id=st.session_state.room_id,
        session_id=st.session_state.session_id,
        sender_id=st.session_state.username,
        sender_name=st.session_state.name,
        role="user",
        message=user_input,
    )

    room_messages = load_room_messages()
    student_history = build_student_history(room_messages)

    refresh_room_state()
    expected_phase_before = st.session_state.expected_phase
    condition = st.session_state.condition

    with st.spinner("正在分析小組討論狀態……"):
        phase_result = analyze_current_phase(
            student_history=student_history,
            expected_phase_before=expected_phase_before,
        )

        quality_result = {
            "triggered": False,
            "trigger_type": "none",
            "intervention": "",
            "reason": "此實驗條件不執行品質介入分析。",
        }

        if condition in {"quality", "hybrid"}:
            quality_result = analyze_current_quality(
                student_history=student_history,
            )

    (
        observed_phase,
        phase_completed,
        next_expected_phase,
        phase_reason,
        phase_confidence,
        shared_evidence,
        evidence_insufficient,
    ) = validate_phase_result(phase_result, expected_phase_before)

    (
        quality_triggered,
        quality_trigger_type,
        quality_intervention,
        quality_reason,
    ) = validate_quality_result(quality_result)

    expected_phase_after = update_expected_phase(
        expected_phase_before=expected_phase_before,
        observed_phase=observed_phase,
        phase_completed=phase_completed,
        next_expected_phase=next_expected_phase,
    )

    structure_triggered = determine_structure_intervention(
        observed_phase=observed_phase,
        expected_phase_before=expected_phase_before,
        student_history=student_history,
        confidence=phase_confidence,
        evidence_insufficient=evidence_insufficient,
    )

    with st.spinner("正在判斷是否需要學習提示……"):
        (
            should_intervene,
            trigger_source,
            trigger_type,
            intervention_message,
        ) = decide_intervention(
            condition=condition,
            structure_triggered=structure_triggered,
            quality_triggered=quality_triggered,
            quality_trigger_type=quality_trigger_type,
            quality_intervention=quality_intervention,
            expected_phase_before=expected_phase_before,
            observed_phase=observed_phase,
            student_history=student_history,
        )

    extended_phase_reason = (
        f"{phase_reason}｜confidence={phase_confidence:.2f}｜"
        f"shared_evidence={shared_evidence}｜"
        f"evidence_insufficient={evidence_insufficient}"
    )

    save_student_record(
        user_input=user_input,
        observed_phase=observed_phase,
        expected_phase_before=expected_phase_before,
        expected_phase_after=expected_phase_after,
        phase_completed=phase_completed,
        quality_triggered=quality_triggered,
        quality_trigger_type=quality_trigger_type,
        should_intervene=should_intervene,
        trigger_source=trigger_source,
        trigger_type=trigger_type,
        phase_reason=extended_phase_reason,
        quality_reason=quality_reason,
        intervention_message=intervention_message,
    )

    if should_intervene and intervention_message:
        save_room_message(
            room_id=st.session_state.room_id,
            session_id=st.session_state.session_id,
            sender_id="ai",
            sender_name="AI助理",
            role="assistant",
            message=intervention_message,
        )


def show_chat_page() -> None:
    if not st.session_state.room_state_initialized:
        try:
            refresh_room_state()
        except Exception as error:
            st.error(f"無法讀取聊天室狀態：{error}")
            return

    st.title("🤖 SSRL AI Assistant")

    top_left, top_right = st.columns([3, 1])

    with top_left:
        st.write(f"歡迎，{st.session_state.name}")

    with top_right:
        if st.button("登出", use_container_width=True):
            logout()
            st.rerun()

    if st.button("🔄 開始新討論", use_container_width=True):
        try:
            begin_new_discussion()
        except Exception as error:
            st.error(f"無法開始新討論：{error}")
            return

        st.success("已為整個小組開始新的討論。")
        st.rerun()

    st.caption(
        f"聊天室：{st.session_state.room_id}｜"
        f"本次討論編號：{st.session_state.session_id}"
    )

    st.divider()
    render_shared_chat()

    user_input = st.chat_input("請輸入討論內容……")

    if not user_input:
        return

    user_input = user_input.strip()
    if not user_input:
        return

    try:
        process_user_message(user_input)
    except Exception as error:
        st.error(f"訊息處理失敗：{error}")
        return

    st.rerun()


if st.session_state.login:
    show_chat_page()
else:
    show_login_page()