from typing import Any, Optional

from google_sheet import get_users_worksheet


# ==========================
# 判斷帳號是否啟用
# ==========================

def is_account_enabled(
    value: Any,
) -> bool:
    """
    判斷 Google Sheet 中的 enabled 欄位
    是否代表帳號已啟用。

    支援：
    TRUE、YES、Y、1、ON

    若 enabled 欄位空白，預設為啟用。
    """

    if value is None:
        return True

    normalized_value = str(
        value
    ).strip().upper()

    if normalized_value == "":
        return True

    return normalized_value in {
        "TRUE",
        "YES",
        "Y",
        "1",
        "ON",
    }


# ==========================
# 驗證登入
# ==========================

def check_login(
    username: str,
    password: str,
) -> Optional[dict]:
    """
    從 Google Sheet 的 Users 工作表驗證登入資料。

    登入成功時回傳：

    {
        "username": "group01",
        "name": "第一組",
        "condition": "structure"
    }

    登入失敗、帳號停用或帳密錯誤時，
    回傳 None。
    """

    username = str(
        username
    ).strip()

    password = str(
        password
    ).strip()

    if not username or not password:
        return None

    try:
        worksheet = get_users_worksheet()

        users = worksheet.get_all_records(
            default_blank="",
        )

    except Exception as error:
        raise RuntimeError(
            f"無法讀取 Google Sheet 的 Users 工作表：{error}"
        ) from error

    for user in users:
        saved_username = str(
            user.get(
                "username",
                "",
            )
        ).strip()

        saved_password = str(
            user.get(
                "password",
                "",
            )
        ).strip()

        saved_name = str(
            user.get(
                "name",
                saved_username,
            )
        ).strip()

        saved_condition = str(
            user.get(
                "condition",
                "",
            )
        ).strip().lower()

        enabled_value = user.get(
            "enabled",
            "",
        )

        if not is_account_enabled(
            enabled_value
        ):
            continue

        if (
            saved_username == username
            and saved_password == password
        ):
            return {
                "username": saved_username,
                "name": saved_name or saved_username,
                "condition": saved_condition,
            }

    return None