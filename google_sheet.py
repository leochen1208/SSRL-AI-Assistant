import os
import time
import uuid
from datetime import datetime
from typing import Any, Callable, TypeVar
from zoneinfo import ZoneInfo

import gspread
import streamlit as st
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials


load_dotenv()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

SPREADSHEET_NAME = "SSRL_AI_Chat_Log"

# 若聊天分析紀錄放在第一張工作表，維持 None。
CHAT_WORKSHEET_NAME = None

USERS_WORKSHEET_NAME = "Users"
ROOM_MESSAGES_WORKSHEET_NAME = "ChatMessages"
ROOM_STATE_WORKSHEET_NAME = "RoomState"

ROOM_MESSAGES_HEADERS = [
    "message_id",
    "room_id",
    "session_id",
    "timestamp",
    "sender_id",
    "sender_name",
    "role",
    "message",
]

ROOM_STATE_HEADERS = [
    "room_id",
    "session_id",
    "condition",
    "expected_phase",
    "updated_at",
]

# 聊天室訊息快取秒數。
# 多個瀏覽器會共用 Streamlit 的應用程式快取，
# 可大幅減少 Google Sheets API 讀取次數。
ROOM_MESSAGES_CACHE_TTL = 8

# RoomState 不需要每兩秒讀取，可稍微快取久一點。
ROOM_STATE_CACHE_TTL = 20

# Users 帳號資料通常不會頻繁變更。
USERS_CACHE_TTL = 60

T = TypeVar("T")


def taipei_now() -> str:
    """回傳臺北時區的目前時間。"""

    return datetime.now(
        ZoneInfo("Asia/Taipei")
    ).strftime("%Y-%m-%d %H:%M:%S")


def normalize_cell_value(value: Any) -> str:
    """將寫入試算表的資料轉成一致格式。"""

    if value is None:
        return ""

    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"

    return str(value).strip()


def is_quota_error(error: Exception) -> bool:
    """判斷是否為 Google Sheets API 429 配額錯誤。"""

    if not isinstance(error, gspread.exceptions.APIError):
        return False

    try:
        status_code = error.response.status_code
        if status_code == 429:
            return True
    except Exception:
        pass

    error_text = str(error).lower()

    return (
        "429" in error_text
        or "quota exceeded" in error_text
        or "rate limit" in error_text
    )


def execute_with_backoff(
    operation: Callable[[], T],
    max_attempts: int = 4,
) -> T:
    """
    執行 Google API 操作。

    遇到 429 時使用指數退避：
    第一次失敗等 2 秒，之後等 4、8 秒。
    """

    for attempt in range(max_attempts):
        try:
            return operation()
        except gspread.exceptions.APIError as error:
            if not is_quota_error(error):
                raise

            if attempt >= max_attempts - 1:
                raise RuntimeError(
                    "Google Sheets API 讀取次數暫時超過限制。"
                    "請稍後再試，並避免頻繁重新整理頁面。"
                ) from error

            wait_seconds = 2 ** (attempt + 1)
            time.sleep(wait_seconds)

    raise RuntimeError("Google Sheets API 操作失敗。")


def get_google_oauth_settings() -> dict:
    """取得 Google OAuth 設定。"""

    local_settings = {
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "refresh_token": os.getenv("GOOGLE_REFRESH_TOKEN"),
        "token_uri": os.getenv(
            "GOOGLE_TOKEN_URI",
            "https://oauth2.googleapis.com/token",
        ),
    }

    required_fields = [
        "client_id",
        "client_secret",
        "refresh_token",
    ]

    if all(
        local_settings.get(field)
        for field in required_fields
    ):
        return {
            key: value.strip()
            if isinstance(value, str)
            else value
            for key, value in local_settings.items()
        }

    try:
        cloud_settings = dict(
            st.secrets["google_oauth"]
        )
    except Exception as error:
        missing_local_fields = [
            field
            for field in required_fields
            if not local_settings.get(field)
        ]

        raise RuntimeError(
            "找不到完整的 Google OAuth 設定。"
            "本機請在 .env 設定 GOOGLE_CLIENT_ID、"
            "GOOGLE_CLIENT_SECRET、GOOGLE_REFRESH_TOKEN；"
            "Streamlit Cloud 請設定 [google_oauth]。"
            f"目前本機缺少：{', '.join(missing_local_fields)}"
        ) from error

    missing_cloud_fields = [
        field
        for field in required_fields
        if not cloud_settings.get(field)
    ]

    if missing_cloud_fields:
        raise RuntimeError(
            "Streamlit Cloud 的 Google OAuth Secrets 缺少欄位："
            + ", ".join(missing_cloud_fields)
        )

    cloud_settings.setdefault(
        "token_uri",
        "https://oauth2.googleapis.com/token",
    )

    return {
        key: value.strip()
        if isinstance(value, str)
        else value
        for key, value in cloud_settings.items()
    }


@st.cache_resource(show_spinner=False)
def get_google_credentials() -> Credentials:
    """
    使用 OAuth refresh token 建立 Google Credentials。

    cache_resource 可避免每次 Streamlit rerun
    都重新刷新 OAuth token。
    """

    oauth_settings = get_google_oauth_settings()

    credentials = Credentials(
        token=None,
        refresh_token=oauth_settings["refresh_token"],
        token_uri=oauth_settings["token_uri"],
        client_id=oauth_settings["client_id"],
        client_secret=oauth_settings["client_secret"],
        scopes=SCOPES,
    )

    try:
        credentials.refresh(Request())
    except Exception as error:
        raise RuntimeError(
            "Google OAuth 憑證更新失敗。"
            "請檢查 OAuth 設定與 refresh token。"
        ) from error

    return credentials


@st.cache_resource(show_spinner=False)
def connect_google_sheet() -> gspread.Client:
    """
    建立並快取 gspread Client。

    所有瀏覽器工作階段共用同一個 Client，
    避免每次重跑程式都重新授權。
    """

    return gspread.authorize(
        get_google_credentials()
    )


@st.cache_resource(show_spinner=False)
def get_spreadsheet() -> gspread.Spreadsheet:
    """取得並快取 Google Spreadsheet。"""

    client = connect_google_sheet()

    try:
        return execute_with_backoff(
            lambda: client.open(SPREADSHEET_NAME)
        )
    except gspread.SpreadsheetNotFound as error:
        raise RuntimeError(
            f"找不到 Google 試算表：{SPREADSHEET_NAME}。"
            "請確認名稱與存取權限。"
        ) from error


def initialize_worksheet(
    worksheet_name: str,
    headers: list[str],
    rows: int = 1000,
    cols: int | None = None,
) -> gspread.Worksheet:
    """
    取得工作表；若不存在則建立。

    這個函式只會由 cache_resource 包裝的函式呼叫，
    不會在每次聊天室同步時重複檢查欄位。
    """

    spreadsheet = get_spreadsheet()

    try:
        worksheet = execute_with_backoff(
            lambda: spreadsheet.worksheet(
                worksheet_name
            )
        )
    except gspread.WorksheetNotFound:
        worksheet = execute_with_backoff(
            lambda: spreadsheet.add_worksheet(
                title=worksheet_name,
                rows=rows,
                cols=cols or max(
                    len(headers),
                    10,
                ),
            )
        )

        execute_with_backoff(
            lambda: worksheet.append_row(
                headers,
                value_input_option="RAW",
            )
        )

        return worksheet

    first_row = execute_with_backoff(
        lambda: worksheet.row_values(1)
    )

    if not first_row:
        execute_with_backoff(
            lambda: worksheet.append_row(
                headers,
                value_input_option="RAW",
            )
        )
    elif first_row != headers:
        raise RuntimeError(
            f"{worksheet_name} 工作表的第一列欄位不正確。"
            f"請改成：{', '.join(headers)}"
        )

    return worksheet


@st.cache_resource(show_spinner=False)
def get_worksheet() -> gspread.Worksheet:
    """取得研究分析紀錄工作表。"""

    spreadsheet = get_spreadsheet()

    if CHAT_WORKSHEET_NAME:
        try:
            return execute_with_backoff(
                lambda: spreadsheet.worksheet(
                    CHAT_WORKSHEET_NAME
                )
            )
        except gspread.WorksheetNotFound as error:
            raise RuntimeError(
                f"找不到聊天紀錄工作表："
                f"{CHAT_WORKSHEET_NAME}。"
            ) from error

    return spreadsheet.sheet1


@st.cache_resource(show_spinner=False)
def get_users_worksheet() -> gspread.Worksheet:
    """取得並快取 Users 工作表。"""

    spreadsheet = get_spreadsheet()

    try:
        return execute_with_backoff(
            lambda: spreadsheet.worksheet(
                USERS_WORKSHEET_NAME
            )
        )
    except gspread.WorksheetNotFound as error:
        raise RuntimeError(
            f"找不到登入工作表："
            f"{USERS_WORKSHEET_NAME}。"
        ) from error


@st.cache_resource(show_spinner=False)
def get_room_messages_worksheet() -> gspread.Worksheet:
    """取得並快取 ChatMessages 工作表。"""

    return initialize_worksheet(
        ROOM_MESSAGES_WORKSHEET_NAME,
        ROOM_MESSAGES_HEADERS,
    )


@st.cache_resource(show_spinner=False)
def get_room_state_worksheet() -> gspread.Worksheet:
    """取得並快取 RoomState 工作表。"""

    return initialize_worksheet(
        ROOM_STATE_WORKSHEET_NAME,
        ROOM_STATE_HEADERS,
        rows=200,
    )


@st.cache_data(
    ttl=USERS_CACHE_TTL,
    show_spinner=False,
)
def get_users_records() -> list[dict]:
    """
    讀取 Users 工作表資料。

    auth.py 若目前直接呼叫
    get_users_worksheet().get_all_records()，
    建議改成呼叫此函式。
    """

    worksheet = get_users_worksheet()

    return execute_with_backoff(
        lambda: worksheet.get_all_records(
            default_blank=""
        )
    )


@st.cache_data(
    ttl=ROOM_STATE_CACHE_TTL,
    show_spinner=False,
)
def get_all_room_states() -> list[dict]:
    """讀取並快取所有 RoomState 紀錄。"""

    worksheet = get_room_state_worksheet()

    return execute_with_backoff(
        lambda: worksheet.get_all_records(
            default_blank=""
        )
    )


def _find_room_state_record(
    room_id: str,
) -> tuple[int | None, dict | None]:
    """
    從快取的 RoomState 資料尋找聊天室。

    row_number 從第 2 列開始，
    因為第 1 列是欄位名稱。
    """

    normalized_room_id = normalize_cell_value(
        room_id
    )

    records = get_all_room_states()

    for row_number, record in enumerate(
        records,
        start=2,
    ):
        record_room_id = str(
            record.get("room_id", "")
        ).strip()

        if record_room_id == normalized_room_id:
            return row_number, record

    return None, None


def get_or_create_room_state(
    room_id: str,
    condition: str,
) -> dict:
    """取得聊天室共用狀態；首次使用時自動建立。"""

    room_id = normalize_cell_value(room_id)
    condition = normalize_cell_value(
        condition
    ).lower()

    if not room_id:
        raise ValueError("room_id 不可為空白。")

    worksheet = get_room_state_worksheet()

    _, record = _find_room_state_record(
        room_id
    )

    if record is not None:
        saved_condition = str(
            record.get("condition", "")
        ).strip().lower()

        if (
            saved_condition
            and saved_condition != condition
        ):
            raise RuntimeError(
                f"聊天室 {room_id} 已設定為 "
                f"{saved_condition}，"
                f"但目前帳號是 {condition}。"
                "同一聊天室的所有帳號必須使用相同 condition。"
            )

        session_id = str(
            record.get("session_id", "")
        ).strip()

        expected_phase = str(
            record.get(
                "expected_phase",
                "task_understanding",
            )
        ).strip() or "task_understanding"

        # 若舊資料沒有 session_id，自動補建。
        if not session_id:
            session_id = str(uuid.uuid4())[:8]

            row_number, _ = _find_room_state_record(
                room_id
            )

            if row_number is not None:
                execute_with_backoff(
                    lambda: worksheet.update(
                        range_name=(
                            f"B{row_number}:E{row_number}"
                        ),
                        values=[[
                            session_id,
                            saved_condition or condition,
                            expected_phase,
                            taipei_now(),
                        ]],
                        value_input_option="USER_ENTERED",
                    )
                )

                get_all_room_states.clear()

        return {
            "room_id": room_id,
            "session_id": session_id,
            "condition": (
                saved_condition or condition
            ),
            "expected_phase": expected_phase,
        }

    session_id = str(uuid.uuid4())[:8]

    execute_with_backoff(
        lambda: worksheet.append_row(
            [
                room_id,
                session_id,
                condition,
                "task_understanding",
                taipei_now(),
            ],
            value_input_option="USER_ENTERED",
        )
    )

    # 新增資料後清除 RoomState 快取，
    # 讓其他瀏覽器可以讀到新聊天室。
    get_all_room_states.clear()

    return {
        "room_id": room_id,
        "session_id": session_id,
        "condition": condition,
        "expected_phase": "task_understanding",
    }


def update_room_expected_phase(
    room_id: str,
    expected_phase: str,
) -> None:
    """更新聊天室共用的預期 SSRL 階段。"""

    room_id = normalize_cell_value(room_id)
    expected_phase = normalize_cell_value(
        expected_phase
    )

    worksheet = get_room_state_worksheet()

    row_number, record = _find_room_state_record(
        room_id
    )

    if row_number is None or record is None:
        # 快取中可能尚未包含剛建立的資料，
        # 清除後重新讀取一次。
        get_all_room_states.clear()

        row_number, record = _find_room_state_record(
            room_id
        )

    if row_number is None or record is None:
        raise RuntimeError(
            f"找不到聊天室 {room_id} 的 RoomState。"
        )

    execute_with_backoff(
        lambda: worksheet.update(
            range_name=(
                f"D{row_number}:E{row_number}"
            ),
            values=[[
                expected_phase,
                taipei_now(),
            ]],
            value_input_option="USER_ENTERED",
        )
    )

    get_all_room_states.clear()


def start_new_room_discussion(
    room_id: str,
    condition: str,
) -> dict:
    """為整個聊天室建立新 session，舊訊息保留。"""

    room_id = normalize_cell_value(room_id)
    condition = normalize_cell_value(
        condition
    ).lower()

    worksheet = get_room_state_worksheet()

    row_number, record = _find_room_state_record(
        room_id
    )

    new_session_id = str(uuid.uuid4())[:8]

    if row_number is None or record is None:
        get_all_room_states.clear()

        row_number, record = _find_room_state_record(
            room_id
        )

    if row_number is None or record is None:
        execute_with_backoff(
            lambda: worksheet.append_row(
                [
                    room_id,
                    new_session_id,
                    condition,
                    "task_understanding",
                    taipei_now(),
                ],
                value_input_option="USER_ENTERED",
            )
        )
    else:
        saved_condition = str(
            record.get("condition", "")
        ).strip().lower()

        if (
            saved_condition
            and saved_condition != condition
        ):
            raise RuntimeError(
                f"聊天室 {room_id} 的 condition 不一致。"
            )

        execute_with_backoff(
            lambda: worksheet.update(
                range_name=(
                    f"B{row_number}:E{row_number}"
                ),
                values=[[
                    new_session_id,
                    condition,
                    "task_understanding",
                    taipei_now(),
                ]],
                value_input_option="USER_ENTERED",
            )
        )

    # 新 session 建立後，清除狀態與聊天快取。
    get_all_room_states.clear()
    get_room_messages.clear()

    return {
        "room_id": room_id,
        "session_id": new_session_id,
        "condition": condition,
        "expected_phase": "task_understanding",
    }


def save_room_message(
    room_id: str,
    session_id: str,
    sender_id: str,
    sender_name: str,
    role: str,
    message: str,
    message_id: str | None = None,
) -> str:
    """將一則多人聊天室訊息寫入 ChatMessages 工作表。"""

    worksheet = get_room_messages_worksheet()

    final_message_id = (
        message_id or str(uuid.uuid4())
    )

    row_data = [
        final_message_id,
        normalize_cell_value(room_id),
        normalize_cell_value(session_id),
        taipei_now(),
        normalize_cell_value(sender_id),
        normalize_cell_value(sender_name),
        normalize_cell_value(role),
        normalize_cell_value(message),
    ]

    try:
        execute_with_backoff(
            lambda: worksheet.append_row(
                row_data,
                value_input_option="USER_ENTERED",
            )
        )
    except gspread.exceptions.APIError as error:
        raise RuntimeError(
            "寫入 ChatMessages 失敗。"
            "請檢查 Google API 配額、"
            "試算表權限或網路連線。"
        ) from error

    # 新訊息寫入後清除聊天室快取，
    # 讓所有瀏覽器下一次同步時取得最新內容。
    get_room_messages.clear()

    return final_message_id


@st.cache_data(
    ttl=ROOM_MESSAGES_CACHE_TTL,
    show_spinner=False,
)
def get_room_messages(
    room_id: str,
    session_id: str,
) -> list[dict]:
    """
    取得指定聊天室與指定討論場次的全部訊息。

    結果會快取數秒。多個瀏覽器若同時讀取相同
    room_id 與 session_id，會共用同一份快取。
    """

    room_id = normalize_cell_value(room_id)
    session_id = normalize_cell_value(
        session_id
    )

    worksheet = get_room_messages_worksheet()

    records = execute_with_backoff(
        lambda: worksheet.get_all_records(
            default_blank=""
        )
    )

    messages = []

    for record in records:
        record_room_id = str(
            record.get("room_id", "")
        ).strip()

        record_session_id = str(
            record.get("session_id", "")
        ).strip()

        if record_room_id != room_id:
            continue

        if record_session_id != session_id:
            continue

        role = str(
            record.get("role", "")
        ).strip()

        if role not in {
            "user",
            "assistant",
        }:
            continue

        content = str(
            record.get("message", "")
        ).strip()

        if not content:
            continue

        messages.append({
            "message_id": str(
                record.get("message_id", "")
            ).strip(),
            "room_id": room_id,
            "session_id": session_id,
            "timestamp": str(
                record.get("timestamp", "")
            ).strip(),
            "sender_id": str(
                record.get("sender_id", "")
            ).strip(),
            "sender_name": str(
                record.get("sender_name", "")
            ).strip(),
            "role": role,
            "content": content,
        })

    return messages


def save_chat(
    student_id,
    session_id,
    condition,
    role,
    message,
    observed_phase,
    expected_phase_before,
    expected_phase_after,
    phase_completed,
    quality_triggered,
    quality_trigger_type,
    should_intervene,
    trigger_source,
    trigger_type,
    phase_reason,
    quality_reason,
    intervention,
) -> bool:
    """
    將學生訊息、階段判斷、品質判斷與介入結果
    寫入研究分析工作表。
    """

    worksheet = get_worksheet()

    row_data = [
        normalize_cell_value(student_id),
        normalize_cell_value(session_id),
        taipei_now(),
        normalize_cell_value(condition),
        normalize_cell_value(role),
        normalize_cell_value(message),
        normalize_cell_value(observed_phase),
        normalize_cell_value(expected_phase_before),
        normalize_cell_value(expected_phase_after),
        normalize_cell_value(phase_completed),
        normalize_cell_value(quality_triggered),
        normalize_cell_value(quality_trigger_type),
        normalize_cell_value(should_intervene),
        normalize_cell_value(trigger_source),
        normalize_cell_value(trigger_type),
        normalize_cell_value(phase_reason),
        normalize_cell_value(quality_reason),
        normalize_cell_value(intervention),
    ]

    try:
        execute_with_backoff(
            lambda: worksheet.append_row(
                row_data,
                value_input_option="USER_ENTERED",
            )
        )
    except gspread.exceptions.APIError as error:
        raise RuntimeError(
            "寫入 Google Sheet 失敗，"
            "請檢查 Google API 權限、"
            "試算表存取權限或網路連線。"
        ) from error

    return True


def clear_google_sheet_caches() -> None:
    """
    手動清除全部 Google Sheet 快取。

    通常不需要呼叫；若你直接手動修改 Google Sheet，
    並希望程式立即重新讀取，可呼叫此函式。
    """

    get_users_records.clear()
    get_all_room_states.clear()
    get_room_messages.clear()


if __name__ == "__main__":
    print("請從 Streamlit 執行此專案。")
