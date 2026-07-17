import json
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from prompt_loader import load_prompt


# ==========================
# 讀取本機環境變數
# ==========================

load_dotenv()


# ==========================
# 取得OpenAI API Key
# ==========================

def get_openai_api_key() -> str:
    """
    取得OpenAI API Key。

    讀取順序：
    1. 本機.env或作業系統環境變數
    2. Streamlit Cloud的st.secrets
    """

    # 本機開發環境
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        return api_key.strip()

    # Streamlit Cloud
    try:
        api_key = st.secrets["OPENAI_API_KEY"]

        if isinstance(api_key, str) and api_key.strip():
            return api_key.strip()

    except (KeyError, FileNotFoundError):
        pass

    raise RuntimeError(
        "找不到OPENAI_API_KEY。"
        "本機請檢查.env檔案；"
        "Streamlit Cloud請檢查App Secrets。"
    )


# ==========================
# 建立OpenAI Client
# ==========================

client = OpenAI(
    api_key=get_openai_api_key()
)


# ==========================
# 載入研究者設定的SSRL Prompt
# ==========================

SSRL_prompt = load_prompt()


# ==========================
# SSRL階段順序
# ==========================

PHASE_SEQUENCE = [
    "task_understanding",
    "planning",
    "monitoring",
    "reflection"
]


# ==========================
# 有效階段集合
# ==========================

VALID_EXPECTED_PHASES = {
    "task_understanding",
    "planning",
    "monitoring",
    "reflection"
}

VALID_OBSERVED_PHASES = {
    "task_understanding",
    "planning",
    "monitoring",
    "reflection",
    "general"
}


# ==========================
# 清理學生討論紀錄
# ==========================

def clean_student_history(student_history):

    if not isinstance(student_history, list):
        raise TypeError(
            "student_history必須是訊息列表。"
        )

    valid_history = []

    for item in student_history:

        if not isinstance(item, dict):
            continue

        content = item.get("content")

        if not isinstance(content, str):
            continue

        content = content.strip()

        if not content:
            continue

        valid_history.append(
            {
                "role": "user",
                "content": content
            }
        )

    return valid_history


# ==========================
# 取得下一個SSRL階段
# ==========================

def get_next_phase(current_phase):

    if current_phase not in PHASE_SEQUENCE:
        return "task_understanding"

    current_index = PHASE_SEQUENCE.index(
        current_phase
    )

    # reflection已經是最後階段
    if current_index >= len(PHASE_SEQUENCE) - 1:
        return "reflection"

    return PHASE_SEQUENCE[
        current_index + 1
    ]


# ==========================
# 預設SSRL分析結果
# ==========================

def get_default_analysis_result(
    expected_phase,
    reason
):

    return {
        "observed_phase": "general",
        "phase_completed": False,
        "next_expected_phase": expected_phase,
        "reason": reason
    }


# ==========================
# 分析SSRL狀態
# ==========================

def analyze_ssrl_state(
    student_history,
    expected_phase
):

    valid_history = clean_student_history(
        student_history
    )

    if expected_phase not in VALID_EXPECTED_PHASES:
        expected_phase = "task_understanding"

    if not valid_history:

        return get_default_analysis_result(
            expected_phase=expected_phase,
            reason="尚無學生討論內容。"
        )

    next_phase = get_next_phase(
        expected_phase
    )

    analysis_prompt = f"""
你是一個社會共享調節學習（SSRL）狀態分析器。

你的任務是分析本次討論紀錄，判斷：

1. 最新一則學生發言主要屬於哪個SSRL階段。
2. 小組是否已經完成系統目前要求的階段。
3. 如果完成，目前應該進入哪個下一階段。

目前系統要求的階段是：

{expected_phase}

如果此階段完成，下一個階段是：

{next_phase}


【SSRL階段定義】

task_understanding：
確認任務要求、問題內容、任務目標，
釐清成員理解，並形成共同理解。

planning：
設定共同目標、討論策略、工作流程、
時間安排或成員分工。

monitoring：
實際執行任務、分享或檢查進度、
發現問題、檢查成果或調整方法。

reflection：
回顧成果、評估目標達成情形、
評估合作過程或提出改進方式。

general：
最新發言過短、內容模糊、單純聊天，
或無法判斷屬於哪個階段。


【階段完成判斷】

task_understanding完成：
討論中已出現對任務要求、問題內容或任務目的的明確理解，
並有共同確認、回應或一致理解的證據。

planning完成：
討論中已形成共同目標，
並至少形成一項具體策略、流程、時間安排或分工。

monitoring完成：
小組已實際執行任務，
並且已檢查進度、成果、問題或進行方法調整。

reflection完成：
小組已對成果或合作歷程進行評估，
或提出具體改進與後續做法。


【重要規則】

1. observed_phase只判斷最新一則學生發言的主要內容。

2. phase_completed必須判斷目前要求的階段
   「{expected_phase}」是否已經完成。

3. 判斷phase_completed時，
   必須閱讀本次討論的完整紀錄，
   不能只閱讀最新一句。

4. 提到某個階段，不代表該階段已完成。

5. 只有出現明確、具體的完成證據，
   phase_completed才可以是true。

6. 不可以因為學生直接討論後續階段，
   就假設前一階段已經完成。

7. 如果目前階段尚未完成，
   next_expected_phase必須維持：
   {expected_phase}

8. 如果目前階段已完成，
   next_expected_phase才可以是：
   {next_phase}

9. 如果目前階段是reflection，
   完成後next_expected_phase仍然是reflection。

10. reason是提供研究者查看的簡短判斷理由，
    不會直接顯示給學生。


請只輸出JSON，不得輸出Markdown或其他文字。

輸出格式：

{{
    "observed_phase": "planning",
    "phase_completed": false,
    "next_expected_phase": "{expected_phase}",
    "reason": "學生正在討論分工，但尚未形成完整的共同計畫。"
}}
"""

    try:

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": analysis_prompt
                },
                *valid_history
            ],
            response_format={
                "type": "json_object"
            },
            temperature=0
        )

    except Exception as error:

        return get_default_analysis_result(
            expected_phase=expected_phase,
            reason=(
                "OpenAI API分析失敗："
                f"{type(error).__name__}"
            )
        )

    content = response.choices[0].message.content

    if not content:

        return get_default_analysis_result(
            expected_phase=expected_phase,
            reason="模型沒有回傳分析結果。"
        )

    try:

        result = json.loads(
            content
        )

    except (json.JSONDecodeError, TypeError):

        return get_default_analysis_result(
            expected_phase=expected_phase,
            reason="模型回傳內容不是有效JSON。"
        )

    observed_phase = result.get(
        "observed_phase",
        "general"
    )

    if observed_phase not in VALID_OBSERVED_PHASES:
        observed_phase = "general"

    phase_completed = result.get(
        "phase_completed",
        False
    )

    # 確保一定是布林值
    if not isinstance(phase_completed, bool):
        phase_completed = False

    # 不直接相信模型提供的next_expected_phase，
    # 由Python依照固定順序更新。
    if phase_completed:

        next_expected_phase = get_next_phase(
            expected_phase
        )

    else:

        next_expected_phase = expected_phase

    reason = result.get(
        "reason",
        ""
    )

    if not isinstance(reason, str):
        reason = str(reason)

    reason = reason.strip()

    if not reason:
        reason = "模型未提供判斷理由。"

    return {
        "observed_phase": observed_phase,
        "phase_completed": phase_completed,
        "next_expected_phase": next_expected_phase,
        "reason": reason
    }


# ==========================
# 產生階段介入訊息
# ==========================

def ask_ai(
    student_history,
    expected_phase,
    observed_phase
):

    valid_history = clean_student_history(
        student_history
    )

    if not valid_history:
        return ""

    if expected_phase not in VALID_EXPECTED_PHASES:
        expected_phase = "task_understanding"

    if observed_phase not in VALID_OBSERVED_PHASES:
        observed_phase = "general"

    intervention_control = f"""
【本次系統判斷】

學生最新討論階段：
{observed_phase}

學生目前應進行的階段：
{expected_phase}

系統已經確認學生進入尚未開放的後續階段，
因此本次必須介入。

請依照目前應進行的階段，
提出一個簡短問題，引導學生回到該階段。


【各階段介入方向】

task_understanding：
引導學生共同確認任務要求、問題內容或任務目的。

planning：
引導學生設定共同目標，
或討論策略、流程、時間安排與分工。

monitoring：
引導學生檢查目前進度、執行情形、
遇到的問題或是否需要調整方法。

reflection：
引導學生回顧成果、評估合作過程，
或提出可以改進的地方。


【輸出限制】

1. 僅輸出給學生看的介入文字。
2. 僅輸出1至2句。
3. 語氣中性。
4. 最多提出一個問題。
5. 不提供任務答案。
6. 不評價學生想法品質。
7. 不解釋SSRL理論。
8. 不輸出階段名稱。
9. 不輸出判斷理由。
10. 不輸出NO_INTERVENTION。
"""

    try:

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": SSRL_prompt
                },
                {
                    "role": "system",
                    "content": intervention_control
                },
                *valid_history
            ],
            temperature=0
        )

    except Exception:

        return get_default_intervention(
            expected_phase
        )

    content = response.choices[0].message.content

    if not content:

        return get_default_intervention(
            expected_phase
        )

    content = content.strip()

    if not content:

        return get_default_intervention(
            expected_phase
        )

    return content


# ==========================
# 預設備援介入文字
# ==========================

def get_default_intervention(expected_phase):

    default_messages = {

        "task_understanding":
            "請先共同確認這項任務要求你們完成什麼。你們對任務內容的理解一致嗎？",

        "planning":
            "請先共同規劃接下來的做法。你們準備如何分工或安排步驟？",

        "monitoring":
            "請先檢查目前的執行情形。現在的進度是否符合原先規劃？",

        "reflection":
            "請一起回顧這次的成果與合作過程。有哪些地方可以改進？"
    }

    return default_messages.get(
        expected_phase,
        ""
    )