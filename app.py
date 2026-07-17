import uuid

import streamlit as st

from auth import check_login
from google_sheet import save_chat
from openai_client import analyze_ssrl_state, ask_ai


# ==========================
# 網頁基本設定
# ==========================

st.set_page_config(
    page_title="SSRL AI Assistant",
    page_icon="🤖",
    layout="centered",
)


# ==========================
# SSRL 階段順序
# ==========================

PHASE_ORDER = {
    "general": 0,
    "task_understanding": 1,
    "planning": 2,
    "monitoring": 3,
    "reflection": 4,
}

VALID_PHASES = {
    "general",
    "task_understanding",
    "planning",
    "monitoring",
    "reflection",
}


# 至少累積幾則學生發言後，才允許 AI 介入。
# 設定為 2，可以避免學生第一句話就被介入。
MIN_MESSAGES_BEFORE_INTERVENTION = 2


# ==========================
# 初始化 Session State
# ==========================

def initialize_session_state() -> None:
    """
    初始化 Streamlit Session State。
    """

    default_values = {
        "login": False,
        "username": "",
        "name": "",
        "condition": "",
        "session_id": str(uuid.uuid4())[:8],
        "messages": [],
        "student_history": [],
        "expected_phase": "task_understanding",
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ==========================
# 重設討論
# ==========================

def reset_discussion() -> None:
    """
    清除目前討論紀錄並建立新的 Session ID。
    """

    st.session_state.messages = []
    st.session_state.student_history = []
    st.session_state.session_id = str(
        uuid.uuid4()
    )[:8]
    st.session_state.expected_phase = (
        "task_understanding"
    )


# ==========================
# 登出
# ==========================

def logout() -> None:
    """
    清除登入狀態與本次討論資料。
    """

    st.session_state.login = False
    st.session_state.username = ""
    st.session_state.name = ""
    st.session_state.condition = ""

    reset_discussion()


# ==========================
# 登入頁面
# ==========================

def show_login_page() -> None:
    """
    顯示登入介面。
    """

    st.title(
        "🤖 SSRL AI Assistant"
    )

    st.write(
        "請輸入帳號與密碼進入討論系統。"
    )

    username = st.text_input(
        "帳號",
        key="login_username",
    )

    password = st.text_input(
        "密碼",
        type="password",
        key="login_password",
    )

    if st.button(
        "登入",
        type="primary",
        use_container_width=True,
    ):
        username = username.strip()
        password = password.strip()

        if not username or not password:
            st.warning(
                "請輸入帳號與密碼。"
            )

            return

        try:
            user = check_login(
                username,
                password,
            )

        except RuntimeError as error:
            st.error(
                str(error)
            )

            return

        except Exception as error:
            st.error(
                f"登入系統發生未預期錯誤：{error}"
            )

            return

        if user is not None:
            st.session_state.login = True
            st.session_state.username = user[
                "username"
            ]
            st.session_state.name = user[
                "name"
            ]
            st.session_state.condition = user.get(
                "condition",
                "",
            )

            reset_discussion()

            st.success(
                "登入成功"
            )

            st.rerun()

        else:
            st.error(
                "帳號或密碼錯誤，或帳號已停用。"
            )


# ==========================
# 分析 SSRL 狀態
# ==========================

def analyze_current_state(
    expected_phase_before: str,
) -> dict:
    """
    呼叫 AI 分析目前討論中的 SSRL 階段。
    """

    try:
        result = analyze_ssrl_state(
            student_history=(
                st.session_state.student_history
            ),
            expected_phase=expected_phase_before,
        )

        if not isinstance(
            result,
            dict,
        ):
            raise TypeError(
                "SSRL 分析結果不是字典格式。"
            )

        return result

    except Exception as error:
        st.warning(
            "階段分析發生問題，"
            "本次暫時記錄為 general。"
        )

        return {
            "observed_phase": "general",
            "phase_completed": False,
            "next_expected_phase": (
                expected_phase_before
            ),
            "reason": f"階段分析失敗：{error}",
        }


# ==========================
# 驗證分析結果
# ==========================

def validate_analysis_result(
    analysis_result: dict,
    expected_phase_before: str,
) -> tuple[str, bool, str, str]:
    """
    整理並驗證 AI 回傳的 SSRL 階段資料。
    """

    observed_phase = analysis_result.get(
        "observed_phase",
        "general",
    )

    phase_completed = analysis_result.get(
        "phase_completed",
        False,
    )

    next_expected_phase = analysis_result.get(
        "next_expected_phase",
        expected_phase_before,
    )

    reason = analysis_result.get(
        "reason",
        "",
    )

    if observed_phase not in VALID_PHASES:
        observed_phase = "general"

    if next_expected_phase not in VALID_PHASES:
        next_expected_phase = (
            expected_phase_before
        )

    # 預期階段不能設定為 general。
    if next_expected_phase == "general":
        next_expected_phase = (
            expected_phase_before
        )

    # 確保 phase_completed 為布林值。
    if not isinstance(
        phase_completed,
        bool,
    ):
        phase_completed = str(
            phase_completed
        ).strip().lower() in {
            "true",
            "1",
            "yes",
        }

    return (
        observed_phase,
        phase_completed,
        next_expected_phase,
        str(reason),
    )


# ==========================
# 判斷是否介入
# ==========================

def determine_intervention(
    observed_phase: str,
    expected_phase: str,
) -> bool:
    """
    判斷學生是否跳過目前應進行的 SSRL 階段。
    """

    message_count = len(
        st.session_state.student_history
    )

    # 無法辨識階段時不介入。
    if observed_phase == "general":
        return False

    # 發言數不足時不介入。
    if (
        message_count
        < MIN_MESSAGES_BEFORE_INTERVENTION
    ):
        return False

    observed_order = PHASE_ORDER.get(
        observed_phase,
        0,
    )

    expected_order = PHASE_ORDER.get(
        expected_phase,
        1,
    )

    # 學生進入尚未開放的後續階段。
    return observed_order > expected_order


# ==========================
# 產生 AI 介入訊息
# ==========================

def generate_intervention(
    expected_phase: str,
    observed_phase: str,
) -> str:
    """
    呼叫 AI 產生 SSRL 階段介入訊息。
    """

    try:
        intervention_message = ask_ai(
            student_history=(
                st.session_state.student_history
            ),
            expected_phase=expected_phase,
            observed_phase=observed_phase,
        )

        if intervention_message is None:
            return ""

        return str(
            intervention_message
        ).strip()

    except Exception as error:
        st.error(
            f"AI 介入訊息產生失敗：{error}"
        )

        return ""


# ==========================
# 寫入 Google Sheet
# ==========================

def save_student_record(
    user_input: str,
    observed_phase: str,
    expected_phase_before: str,
    expected_phase_after: str,
    phase_completed: bool,
    should_intervene: bool,
    reason: str,
    intervention_message: str,
) -> None:
    """
    將學生訊息與 SSRL 分析結果
    寫入 Google Sheet。
    """

    try:
        # 優先使用 username 作為 student_id。
        # 若 username 不存在，才使用 name。
        student_id = (
            st.session_state.username
            or st.session_state.name
        )

        save_chat(
            student_id=student_id,
            session_id=(
                st.session_state.session_id
            ),
            role="user",
            message=user_input,
            observed_phase=observed_phase,
            expected_phase_before=(
                expected_phase_before
            ),
            expected_phase_after=(
                expected_phase_after
            ),
            phase_completed=phase_completed,
            should_intervene=should_intervene,
            reason=reason,
            intervention=intervention_message,
        )

    except Exception as error:
        st.warning(
            f"Google Sheet 紀錄失敗：{error}"
        )


# ==========================
# 顯示聊天主頁
# ==========================

def show_chat_page() -> None:
    """
    顯示 SSRL 聊天介面。
    """

    st.title(
        "🤖 SSRL AI Assistant"
    )

    top_left, top_right = st.columns(
        [3, 1]
    )

    with top_left:
        st.write(
            f"歡迎，{st.session_state.name}"
        )

    with top_right:
        if st.button(
            "登出",
            use_container_width=True,
        ):
            logout()
            st.rerun()

    # ==========================
    # 開始新討論
    # ==========================

    if st.button(
        "🔄 開始新討論",
        use_container_width=True,
    ):
        reset_discussion()
        st.rerun()

    st.caption(
        f"本次討論編號："
        f"{st.session_state.session_id}"
    )

    # 正式實驗若不希望學生看到目前階段，
    # 可以刪除或註解以下內容。
    st.caption(
        f"系統目前預期階段："
        f"{st.session_state.expected_phase}"
    )

    # 正式實驗若不希望學生看到組別條件，
    # 請不要顯示以下內容。
    #
    # st.caption(
    #     f"實驗條件："
    #     f"{st.session_state.condition}"
    # )

    st.divider()

    # ==========================
    # 顯示本次聊天紀錄
    # ==========================

    for message in st.session_state.messages:
        with st.chat_message(
            message["role"]
        ):
            st.write(
                message["content"]
            )

    # ==========================
    # 接收學生輸入
    # ==========================

    user_input = st.chat_input(
        "請輸入討論內容……"
    )

    if not user_input:
        return

    user_input = user_input.strip()

    if not user_input:
        return

    expected_phase_before = (
        st.session_state.expected_phase
    )

    # ==========================
    # 1. 儲存學生訊息
    # ==========================

    student_message = {
        "role": "user",
        "content": user_input,
    }

    st.session_state.messages.append(
        student_message
    )

    st.session_state.student_history.append(
        student_message
    )

    # ==========================
    # 2. 顯示學生訊息
    # ==========================

    with st.chat_message(
        "user"
    ):
        st.write(
            user_input
        )

    # ==========================
    # 3. 分析目前 SSRL 狀態
    # ==========================

    with st.spinner(
        "正在分析小組討論狀態……"
    ):
        analysis_result = analyze_current_state(
            expected_phase_before
        )

    (
        observed_phase,
        phase_completed,
        next_expected_phase,
        reason,
    ) = validate_analysis_result(
        analysis_result,
        expected_phase_before,
    )

    # ==========================
    # 4. 更新預期階段
    # ==========================

    if phase_completed:
        st.session_state.expected_phase = (
            next_expected_phase
        )

    else:
        st.session_state.expected_phase = (
            expected_phase_before
        )

    expected_phase_after = (
        st.session_state.expected_phase
    )

    # ==========================
    # 5. 判斷是否介入
    # ==========================

    should_intervene = determine_intervention(
        observed_phase=observed_phase,
        expected_phase=expected_phase_after,
    )

    # ==========================
    # 6. 產生 AI 介入訊息
    # ==========================

    intervention_message = ""

    if should_intervene:
        with st.spinner(
            "正在產生學習提示……"
        ):
            intervention_message = (
                generate_intervention(
                    expected_phase=(
                        expected_phase_after
                    ),
                    observed_phase=(
                        observed_phase
                    ),
                )
            )

    # ==========================
    # 7. 寫入 Google Sheet
    # ==========================

    save_student_record(
        user_input=user_input,
        observed_phase=observed_phase,
        expected_phase_before=(
            expected_phase_before
        ),
        expected_phase_after=(
            expected_phase_after
        ),
        phase_completed=phase_completed,
        should_intervene=should_intervene,
        reason=reason,
        intervention_message=(
            intervention_message
        ),
    )

    # ==========================
    # 8. 顯示 AI 介入訊息
    # ==========================

    if (
        should_intervene
        and intervention_message
    ):
        assistant_message = {
            "role": "assistant",
            "content": intervention_message,
        }

        st.session_state.messages.append(
            assistant_message
        )

        with st.chat_message(
            "assistant"
        ):
            st.write(
                intervention_message
            )


# ==========================
# 主程式
# ==========================

if st.session_state.login:
    show_chat_page()

else:
    show_login_page()