import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import gspread
import streamlit as st
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials


# ==========================
# 載入本機 .env
# ==========================

load_dotenv()


# ==========================
# Google API 權限
# ==========================

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


# ==========================
# Google Sheet 設定
# ==========================

SPREADSHEET_NAME = "SSRL_AI_Chat_Log"

# 聊天紀錄工作表。
# 設為 None 時，使用試算表中的第一個工作表。
CHAT_WORKSHEET_NAME = None

# 登入帳號工作表名稱。
USERS_WORKSHEET_NAME = "Users"


# ==========================
# 讀取 OAuth 設定
# ==========================

def get_google_oauth_settings() -> dict:
    """
    取得 Google OAuth 設定。

    讀取順序：
    1. 本機 .env
    2. Streamlit Cloud 的 st.secrets["google_oauth"]
    """

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

    local_settings_complete = all(
        local_settings.get(field)
        for field in required_fields
    )

    if local_settings_complete:
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
            "本機請在 .env 設定 "
            "GOOGLE_CLIENT_ID、GOOGLE_CLIENT_SECRET、"
            "GOOGLE_REFRESH_TOKEN；"
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


# ==========================
# 取得 OAuth 憑證
# ==========================

def get_google_credentials() -> Credentials:
    """
    使用 client ID、client secret 與 refresh token
    建立 Google OAuth Credentials。

    access token 過期時，會自動使用 refresh token
    取得新的 access token。
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
        credentials.refresh(
            Request()
        )

    except Exception as error:
        raise RuntimeError(
            "Google OAuth 憑證更新失敗。"
            "請檢查 GOOGLE_CLIENT_ID、"
            "GOOGLE_CLIENT_SECRET 與 "
            "GOOGLE_REFRESH_TOKEN 是否正確，"
            "以及 refresh token 是否仍然有效。"
        ) from error

    return credentials


# ==========================
# 連接 Google Sheet
# ==========================

def connect_google_sheet() -> gspread.Client:
    """
    建立並回傳 gspread Client。
    """

    credentials = get_google_credentials()

    return gspread.authorize(
        credentials
    )


# ==========================
# 取得 Google 試算表
# ==========================

def get_spreadsheet() -> gspread.Spreadsheet:
    """
    開啟並回傳指定的 Google 試算表。
    """

    client = connect_google_sheet()

    try:
        return client.open(
            SPREADSHEET_NAME
        )

    except gspread.SpreadsheetNotFound as error:
        raise RuntimeError(
            f"找不到 Google 試算表：{SPREADSHEET_NAME}。"
            "請確認試算表名稱正確，"
            "並確認 OAuth 授權帳號有權限存取該試算表。"
        ) from error


# ==========================
# 取得聊天紀錄工作表
# ==========================

def get_worksheet() -> gspread.Worksheet:
    """
    取得聊天紀錄工作表。

    CHAT_WORKSHEET_NAME 為 None 時，
    使用試算表中的第一個工作表。
    """

    spreadsheet = get_spreadsheet()

    if CHAT_WORKSHEET_NAME:
        try:
            return spreadsheet.worksheet(
                CHAT_WORKSHEET_NAME
            )

        except gspread.WorksheetNotFound as error:
            raise RuntimeError(
                f"找不到聊天紀錄工作表："
                f"{CHAT_WORKSHEET_NAME}。"
            ) from error

    return spreadsheet.sheet1


# ==========================
# 取得 Users 登入工作表
# ==========================

def get_users_worksheet() -> gspread.Worksheet:
    """
    取得名為 Users 的登入帳號工作表。
    """

    spreadsheet = get_spreadsheet()

    try:
        return spreadsheet.worksheet(
            USERS_WORKSHEET_NAME
        )

    except gspread.WorksheetNotFound as error:
        raise RuntimeError(
            f"找不到登入工作表：{USERS_WORKSHEET_NAME}。"
            "請在 Google 試算表中新增一個名稱完全相同的工作表。"
        ) from error


# ==========================
# 清理寫入資料
# ==========================

def normalize_cell_value(
    value: Any,
) -> str:
    """
    將資料轉換成適合寫入 Google Sheet 的文字。
    """

    if value is None:
        return ""

    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"

    return str(value).strip()


# ==========================
# 儲存聊天與 SSRL 分析紀錄
# ==========================

def save_chat(
    student_id,
    session_id,
    role,
    message,
    observed_phase,
    expected_phase_before,
    expected_phase_after,
    phase_completed,
    should_intervene,
    reason,
    intervention,
) -> bool:
    """
    將學生訊息、SSRL 階段分析與 AI 介入結果
    寫入 Google Sheet。

    欄位順序：
    1. student_id
    2. session_id
    3. timestamp
    4. role
    5. message
    6. observed_phase
    7. expected_phase_before
    8. expected_phase_after
    9. phase_completed
    10. should_intervene
    11. reason
    12. intervention
    """

    worksheet = get_worksheet()

    timestamp = datetime.now(
        ZoneInfo("Asia/Taipei")
    ).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    row_data = [
        normalize_cell_value(student_id),
        normalize_cell_value(session_id),
        timestamp,
        normalize_cell_value(role),
        normalize_cell_value(message),
        normalize_cell_value(observed_phase),
        normalize_cell_value(expected_phase_before),
        normalize_cell_value(expected_phase_after),
        normalize_cell_value(phase_completed),
        normalize_cell_value(should_intervene),
        normalize_cell_value(reason),
        normalize_cell_value(intervention),
    ]

    try:
        worksheet.append_row(
            row_data,
            value_input_option="USER_ENTERED",
        )

    except gspread.exceptions.APIError as error:
        raise RuntimeError(
            "寫入 Google Sheet 失敗，"
            "請檢查 Google API 權限、"
            "試算表存取權限或網路連線。"
        ) from error

    return True


# ==========================
# 本機連線測試
# ==========================

if __name__ == "__main__":
    save_chat(
        student_id="TEST001",
        session_id="TEST_SESSION",
        role="user",
        message="測試 OAuth Refresh Token 連線",
        observed_phase="task_understanding",
        expected_phase_before="task_understanding",
        expected_phase_after="task_understanding",
        phase_completed=False,
        should_intervene=False,
        reason="這是一筆 Google Sheet 連線測試資料。",
        intervention="",
    )

    print(
        "Google Sheet 寫入完成"
    )