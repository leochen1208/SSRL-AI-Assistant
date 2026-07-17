import json
import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from prompt_loader import load_prompt


load_dotenv()

MODEL_NAME = "gpt-4.1-mini"

PHASE_SEQUENCE = [
    "task_understanding",
    "planning",
    "monitoring",
    "reflection",
]

VALID_EXPECTED_PHASES = set(PHASE_SEQUENCE)
VALID_OBSERVED_PHASES = VALID_EXPECTED_PHASES | {"general"}

VALID_QUALITY_TRIGGER_TYPES = {
    "negative_value_monitoring",
    "concept_confusion",
    "lack_of_shared_perspective",
    "none",
}


def get_openai_api_key() -> str:
    """從本機 .env 或 Streamlit Secrets 取得 OpenAI API Key。"""

    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        return api_key.strip()

    try:
        api_key = st.secrets["OPENAI_API_KEY"]

        if isinstance(api_key, str) and api_key.strip():
            return api_key.strip()
    except (KeyError, FileNotFoundError):
        pass

    raise RuntimeError(
        "找不到 OPENAI_API_KEY。"
        "本機請檢查 .env；Streamlit Cloud 請檢查 App Secrets。"
    )


client = OpenAI(api_key=get_openai_api_key())
SSRL_prompt = load_prompt()


def normalize_speaker(value: Any) -> str:
    """清理發言者名稱，避免將空白或過長文字放入模型訊息。"""

    speaker = str(value or "").strip()

    if not speaker:
        return "小組成員"

    return speaker[:50]


def clean_student_history(student_history: list[dict]) -> list[dict]:
    """
    將多人聊天室紀錄整理成 OpenAI 可讀格式。

    支援以下輸入：
    {
        "role": "user",
        "content": "我們先確認題目",
        "speaker": "小明"
    }

    或：
    {
        "role": "user",
        "content": "我們先確認題目",
        "sender_name": "小明"
    }

    每筆訊息會轉成：
    小明：我們先確認題目

    如此模型可以辨識不同組員，而不是將全部發言視為同一位學生。
    """

    if not isinstance(student_history, list):
        raise TypeError("student_history 必須是訊息列表。")

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

        speaker = normalize_speaker(
            item.get("speaker")
            or item.get("sender_name")
            or item.get("name")
        )

        valid_history.append({
            "role": "user",
            "content": f"{speaker}：{content}",
        })

    return valid_history


def get_next_phase(current_phase: str) -> str:
    """取得目前階段完成後的下一個預期階段。"""

    if current_phase not in PHASE_SEQUENCE:
        return "task_understanding"

    current_index = PHASE_SEQUENCE.index(current_phase)

    if current_index >= len(PHASE_SEQUENCE) - 1:
        return "reflection"

    return PHASE_SEQUENCE[current_index + 1]


def get_default_analysis_result(
    expected_phase: str,
    reason: str,
) -> dict:
    return {
        "observed_phase": "general",
        "phase_completed": False,
        "next_expected_phase": expected_phase,
        "reason": reason,
    }


def get_default_quality_result(reason: str) -> dict:
    return {
        "triggered": False,
        "trigger_type": "none",
        "intervention": "",
        "reason": reason,
    }


def parse_json_response(
    response_content: str | None,
) -> dict | None:
    """安全解析模型 JSON 回覆。"""

    try:
        result = json.loads(response_content or "")
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(result, dict):
        return None

    return result


def analyze_ssrl_state(
    student_history: list[dict],
    expected_phase: str,
) -> dict:
    """
    分析整個小組目前主要展現的 SSRL 階段。

    所有 Structure、Quality、Hybrid 組別都會執行此分析。
    """

    valid_history = clean_student_history(student_history)

    if expected_phase not in VALID_EXPECTED_PHASES:
        expected_phase = "task_understanding"

    if not valid_history:
        return get_default_analysis_result(
            expected_phase,
            "尚無小組討論內容。",
        )

    analysis_prompt = f"""
【角色】
你是一個社會共享調節學習（SSRL）階段分析助理。

你的分析單位是「整個小組」，不是單一學生。
student_history 包含同一聊天室中，不同小組成員依時間排列的發言。
每則發言前方會標示發言者，例如「小明：……」。

你只負責：
1. 判斷小組目前主要展現的 SSRL 階段。
2. 判斷小組是否已完成目前預期階段的核心共同調節行為。

你不提供任務答案，也不評估調節品質。

【SSRL 階段】
1. task_understanding
小組成員共同確認、重述、解釋或釐清任務主題、要求、目標、產出或限制。
只有個別成員單方面陳述，且其他成員尚未回應、確認或協商時，不宜直接視為已形成共同任務理解。

2. planning
小組共同設定目標，協商策略、步驟、時間、分工或工作流程。
只有個人宣告自己要做什麼，但沒有與其他成員形成共同安排時，不宜視為完成共同規劃。

3. monitoring
小組實際提出答案、想法、理由、例子或方案；檢查進度、理解、策略效果、問題，並進行修正或調整。
單純提出任務內容、理由、案例或方案，通常判定為 monitoring。

4. reflection
小組回顧或評估已進行或已完成的成果、合作過程或策略效果，並提出改善方式。

5. general
閒聊、簡短附和、拒絕參與、單純困惑、無意義內容、要求直接答案，或資訊不足以判斷 SSRL 階段。

【判斷原則】
- 每次只判定一個最主要階段。
- 判斷整個小組最近主要正在做什麼，而不是只看某一位成員。
- 不要求每一位成員都發言，但必須有可觀察的共同確認、協商、整合或回應。
- 「好」、「同意」、「可以」等簡短回應，可作為共同確認的證據，但需結合前文判斷。
- 單純表示不知道、不想做或要求答案，屬於 general。
- 檢查當下進度或方法屬於 monitoring。
- 回顧已完成成果或整體合作過程屬於 reflection。
- 小組可以回到先前階段補充，仍依當下實際行為分類。
- 資訊不足時使用 general。

【phase_completed 判斷】
目前預期階段是：{expected_phase}

只有在小組已形成該階段的「共同結果」時，才回傳 true。

task_understanding 完成：
小組已形成對任務要求、目的、產出或限制的共同理解。

planning 完成：
小組已形成可執行的共同目標、策略、步驟或分工。

monitoring 完成：
小組已實際推進任務內容，並形成足以進入成果回顧的內容或方案。

reflection 完成：
小組已完成對成果或合作過程的評估，並提出至少一項結論或改善方向。

若最新發言屬於其他階段、general，或只由個別成員提及但尚未形成共同結果，回傳 false。

【next_expected_phase】
請原樣回傳目前預期階段：{expected_phase}
Python 程式會自行更新階段，模型不要自行推進。

請只輸出 JSON：
{{
  "observed_phase": "monitoring",
  "phase_completed": false,
  "next_expected_phase": "{expected_phase}",
  "reason": "簡短說明小組行為與判斷依據。"
}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": analysis_prompt,
                },
                *valid_history,
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
    except Exception as error:
        return get_default_analysis_result(
            expected_phase,
            f"OpenAI API 階段分析失敗：{type(error).__name__}",
        )

    result = parse_json_response(
        response.choices[0].message.content
    )

    if result is None:
        return get_default_analysis_result(
            expected_phase,
            "模型回傳內容不是有效 JSON。",
        )

    observed_phase = result.get(
        "observed_phase",
        "general",
    )

    if observed_phase not in VALID_OBSERVED_PHASES:
        observed_phase = "general"

    phase_completed = result.get(
        "phase_completed",
        False,
    )

    if not isinstance(phase_completed, bool):
        phase_completed = False

    next_expected_phase = (
        get_next_phase(expected_phase)
        if phase_completed
        else expected_phase
    )

    reason = str(
        result.get("reason", "")
    ).strip()

    if not reason:
        reason = "模型未提供階段判斷理由。"

    return {
        "observed_phase": observed_phase,
        "phase_completed": phase_completed,
        "next_expected_phase": next_expected_phase,
        "reason": reason,
    }


def analyze_quality_state(
    student_history: list[dict],
) -> dict:
    """
    分析整個小組的 SSRL 調節品質。

    Quality 與 Hybrid 組使用。
    """

    valid_history = clean_student_history(student_history)

    if len(valid_history) < 3:
        return get_default_quality_result(
            "有效小組發言少於三個回合，尚未檢查品質介入條件。"
        )

    # 最近三回合為主要判斷依據，額外保留前五回合協助理解語境。
    recent_history = valid_history[-8:]

    quality_prompt = """
【角色】
你是一個 SSRL 調節品質助理。

你的分析單位是「整個小組」，不是單一學生。
輸入包含同一聊天室中不同組員依時間排列的發言，
每則發言前方會標示發言者，例如「小明：……」。

你的任務是評估小組是否出現特定的調節品質問題，
並在符合介入條件時提供簡短引導。

你不負責監控 SSRL 階段順序，也不提供任務答案。

【判斷範圍】
以最近三個有效小組發言回合為主要判斷範圍；
更早內容只用來理解語境與確認問題是否已被釐清。

「三個發言回合」可以來自同一位或不同組員。
但你必須分析這些發言是否構成小組層次的持續問題，
不能因某位成員單次表達就推論整組都有問題。

同一輪最多觸發一種條件，必須依下列順序檢查：
1. negative_value_monitoring
2. concept_confusion
3. lack_of_shared_perspective

若前一項成立，不再檢查後面的條件。

【條件一：negative_value_monitoring】
最近連續三個有效發言，都對目前能力、方法、進展或成功可能性表達明確負向評估，
例如做不到、不能完成、沒有辦法、這樣不行、一定會失敗。

以下情況不應觸發：
- 只出現單一「不」字。
- 合理否定某個答案。
- 比較不同方案。
- 修正錯誤。
- 表達不同意，但仍持續討論。
- 一位成員負向表達後，其他成員已提出可行方法或鼓勵。

介入方向：
簡短指出小組遇到困難，再引導描述問題、辨識資源或思考如何取得進展。

【條件二：concept_confusion】
最近三個有效發言持續表達困惑、不理解，
或持續指向同一知識概念、任務要求或方法理解問題，
而且小組尚未自行釐清，也沒有形成可行理解。

以下情況不應觸發：
- 只有一個發言表示困惑。
- 後續成員已提供合理解釋，且小組已確認理解。
- 只是對不同方案進行詢問或澄清，但討論仍有效推進。

介入方向：
指出小組可能仍有困惑，再引導成員說明目前想理解的內容或共同確認理解。

【條件三：lack_of_shared_perspective】
最近三個有效發言持續以個人立場、個人決定或個人任務為中心，
且沒有出現整合觀點、共同目標、共同計畫、協商或相互回應。

不能只看「我／你」的字面詞頻。
必須從語意判斷小組是否缺乏共同調節。

以下情況不應觸發：
- 正常分工，例如「我找資料，你整理」。
- 成員交換不同意見後正在協商。
- 使用「我認為」提出想法，但後續有整合或回應。
- 一句話出現我、你，就直接判定缺乏共享觀點。

介入方向：
指出目前討論較偏向個人觀點，
再引導小組整合觀點、形成共同目標或共同計畫。

【介入格式】
- 只輸出一至二句。
- 最多一個主要問題。
- 語氣中性。
- 使用「你們」稱呼整個小組。
- 不提供任務答案。
- 不評價學生想法正確性。
- 不補充 SSRL 理論。
- 沒有觸發時 intervention 必須是空字串。

請只輸出 JSON：
{
  "triggered": true,
  "trigger_type": "concept_confusion",
  "intervention": "我注意到你們可能還有些困惑。你們目前最需要共同釐清的是什麼？",
  "reason": "最近三個發言持續針對相同內容表達困惑，且尚未形成共同理解。"
}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": quality_prompt,
                },
                *recent_history,
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
    except Exception as error:
        return get_default_quality_result(
            f"OpenAI API 品質分析失敗：{type(error).__name__}"
        )

    result = parse_json_response(
        response.choices[0].message.content
    )

    if result is None:
        return get_default_quality_result(
            "模型回傳的品質分析不是有效 JSON。"
        )

    triggered = result.get(
        "triggered",
        False,
    )

    if not isinstance(triggered, bool):
        triggered = False

    trigger_type = result.get(
        "trigger_type",
        "none",
    )

    if trigger_type not in VALID_QUALITY_TRIGGER_TYPES:
        trigger_type = "none"

    intervention = str(
        result.get("intervention", "")
    ).strip()

    reason = str(
        result.get("reason", "")
    ).strip()

    if not triggered or trigger_type == "none":
        triggered = False
        trigger_type = "none"
        intervention = ""

    if triggered and not intervention:
        intervention = get_default_quality_intervention(
            trigger_type
        )

    if not reason:
        reason = "模型未提供品質判斷理由。"

    return {
        "triggered": triggered,
        "trigger_type": trigger_type,
        "intervention": intervention,
        "reason": reason,
    }


def ask_ai(
    student_history: list[dict],
    expected_phase: str,
    observed_phase: str,
) -> str:
    """
    Structure 或 Hybrid 組發生跳階時，產生小組層次的結構介入訊息。
    """

    valid_history = clean_student_history(student_history)

    if not valid_history:
        return ""

    if expected_phase not in VALID_EXPECTED_PHASES:
        expected_phase = "task_understanding"

    if observed_phase not in VALID_OBSERVED_PHASES:
        observed_phase = "general"

    intervention_control = f"""
【分析單位】
目前對話來自同一小組的多位成員。
每則發言前方標示發言者。
請對整個小組說話，使用「你們」，不要針對單一成員。

【本次系統判斷】
小組最新主要討論階段：{observed_phase}
小組目前應先完成的階段：{expected_phase}

小組已進入較後面的階段，因此本次必須介入。
請提出一個簡短問題，引導整個小組回到目前應完成的階段。

【介入方向】
task_understanding：
引導小組共同確認任務要求、問題內容、任務目的、產出或限制。

planning：
引導小組形成共同目標，或協商策略、流程、時間與分工。

monitoring：
引導小組提出並整理任務內容，或共同檢查進度、問題及方法調整。

reflection：
引導小組共同回顧成果、合作過程、策略效果或可改進之處。

【輸出限制】
- 只輸出給小組成員看的介入文字。
- 一至二句。
- 最多一個主要問題。
- 使用「你們」。
- 語氣中性。
- 不提供答案。
- 不評價想法。
- 不解釋 SSRL。
- 不輸出階段名稱。
- 不輸出系統判斷理由。
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SSRL_prompt,
                },
                {
                    "role": "system",
                    "content": intervention_control,
                },
                *valid_history,
            ],
            temperature=0,
        )
    except Exception:
        return get_default_intervention(
            expected_phase
        )

    content = str(
        response.choices[0].message.content or ""
    ).strip()

    return content or get_default_intervention(
        expected_phase
    )


def get_default_intervention(
    expected_phase: str,
) -> str:
    default_messages = {
        "task_understanding": (
            "請先共同確認這項任務要求你們完成什麼。"
            "你們對任務內容的理解一致嗎？"
        ),
        "planning": (
            "請先共同規劃接下來的做法。"
            "你們準備如何分工或安排步驟？"
        ),
        "monitoring": (
            "請先共同整理目前的任務內容與進度。"
            "你們接下來還需要完成什麼？"
        ),
        "reflection": (
            "請一起回顧這次的成果與合作過程。"
            "你們認為有哪些地方可以改進？"
        ),
    }

    return default_messages.get(
        expected_phase,
        "",
    )


def get_default_quality_intervention(
    trigger_type: str,
) -> str:
    default_messages = {
        "negative_value_monitoring": (
            "我注意到你們遇到了一些困難。"
            "你們可以先共同描述目前發生了什麼嗎？"
        ),
        "concept_confusion": (
            "我注意到你們可能還有些困惑。"
            "你們目前最需要共同釐清的是什麼？"
        ),
        "lack_of_shared_perspective": (
            "我注意到你們目前較偏向個人觀點。"
            "你們可以如何整合成共同計畫？"
        ),
    }

    return default_messages.get(
        trigger_type,
        "",
    )