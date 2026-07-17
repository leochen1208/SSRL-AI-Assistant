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
# 載入本機.env
# ==========================

load_dotenv()


# ==========================
# Google API權限
# ==========================

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


# ==========================
# Google Sheet設定
# ==========================

SPREADSHEET_NAME = "SSRL_AI_Chat_Log"
WORKSHEET_NAME = None

# WORKSHEET_NAME設為None時，使用第一個工作表。
# 如要指定工作表，可改成：
# WORKSHEET_NAME = "工作表1"


# ==========================
# 讀取OAuth設定
# ==========================

def get_google_oauth_settings() -> dict:
    """
    取得Google OAuth設定。

    讀取順序：
    1. 本機.env
    2. Streamlit Cloud的st.secrets["google_oauth"]
    """

    # 先嘗試讀取本機.env
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

    # 本機.env不完整時，改讀取Streamlit Cloud Secrets
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
            "找不到完整的Google OAuth設定。"
            "本機請在.env設定："
            "GOOGLE_CLIENT_ID、GOOGLE_CLIENT_SECRET、"
            "GOOGLE_REFRESH_TOKEN；"
            "Streamlit Cloud請設定[google_oauth]。"
            f"目前本機缺少：{', '.join(missing_local_fields)}"
        ) from error

    missing_cloud_fields = [
        field
        for field in required_fields
        if not cloud_settings.get(field)
    ]

    if missing_cloud_fields:

        raise RuntimeError(
            "Streamlit Cloud的Google OAuth Secrets缺少欄位："
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
# 取得OAuth憑證
# ==========================

def get_google_credentials() -> Credentials:
    """
    使用client ID、client secret與refresh token
    建立Google OAuth Credentials。

    access token過期時，會自動使用refresh token
    取得新的access token。
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
            "Google OAuth憑證更新失敗。"
            "請檢查GOOGLE_CLIENT_ID、"
            "GOOGLE_CLIENT_SECRET與"
            "GOOGLE_REFRESH_TOKEN是否正確，"
            "以及refresh token是否仍然有效。"
        ) from error

    return credentials


# ==========================
# 連接Google Sheet
# ==========================

def connect_google_sheet() -> gspread.Client:
    """
    建立並回傳gspread Client。
    """

    credentials = get_google_credentials()

    return gspread.authorize(
        credentials
    )


# ==========================
# 取得指定工作表
# ==========================

def get_worksheet() -> gspread.Worksheet:
    """
    開啟指定Google試算表並取得工作表。

    WORKSHEET_NAME為None時，
    使用試算表中的第一個工作表。
    """

    client = connect_google_sheet()

    try:

        spreadsheet = client.open(
            SPREADSHEET_NAME
        )

    except gspread.SpreadsheetNotFound as error:

        raise RuntimeError(
            f"找不到Google試算表：{SPREADSHEET_NAME}。"
            "請確認試算表名稱正確，"
            "並確認OAuth授權帳號有權限存取該試算表。"
        ) from error

    if WORKSHEET_NAME:

        try:

            return spreadsheet.worksheet(
                WORKSHEET_NAME
            )

        except gspread.WorksheetNotFound as error:

            raise RuntimeError(
                f"找不到工作表：{WORKSHEET_NAME}。"
            ) from error

    return spreadsheet.sheet1


# ==========================
# 清理寫入資料
# ==========================

def normalize_cell_value(
    value: Any
) -> str:
    """
    將資料轉換成適合寫入Google Sheet的文字。
    """

    if value is None:
        return ""

    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"

    return str(value).strip()


# ==========================
# 儲存聊天與SSRL分析紀錄
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
    將學生訊息、SSRL階段分析與AI介入結果
    寫入Google Sheet。

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
            "寫入Google Sheet失敗，"
            "請檢查Google API權限、"
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
        message="測試OAuth Refresh Token連線",
        observed_phase="task_understanding",
        expected_phase_before="task_understanding",
        expected_phase_after="task_understanding",
        phase_completed=False,
        should_intervene=False,
        reason="這是一筆Google Sheet連線測試資料。",
        intervention="",
    )

    print(
        "Google Sheet寫入完成"
    )