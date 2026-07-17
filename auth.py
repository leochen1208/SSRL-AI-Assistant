from typing import Any, Optional

from google_sheet import get_users_records


ALLOWED_CONDITIONS = {
    "structure",
    "quality",
    "hybrid",
}


def is_account_enabled(value: Any) -> bool:
    """判斷 Google Sheet 中的 enabled 欄位是否代表帳號已啟用。"""

    if value is None:
        return True

    normalized_value = str(value).strip().upper()

    if normalized_value == "":
        return True

    return normalized_value in {
        "TRUE",
        "YES",
        "Y",
        "1",
        "ON",
    }


def check_login(
    username: str,
    password: str,
) -> Optional[dict]:
    """
    從 Google Sheet 的 Users 工作表驗證登入資料。

    Users 工作表至少需要以下欄位：
    username、password、name、condition、room_id、enabled
    """

    username = str(username).strip()
    password = str(password).strip()

    if not username or not password:
        return None

    try:
        # 使用 google_sheet.py 的快取版本，
        # 避免每次登入都直接讀取 Google Sheet。
        users = get_users_records()
    except Exception as error:
        raise RuntimeError(
            f"無法讀取 Google Sheet 的 Users 工作表：{error}"
        ) from error

    for user in users:
        saved_username = str(
            user.get("username", "")
        ).strip()

        saved_password = str(
            user.get("password", "")
        ).strip()

        saved_name = str(
            user.get("name", saved_username)
        ).strip()

        saved_condition = str(
            user.get("condition", "")
        ).strip().lower()

        saved_room_id = str(
            user.get("room_id", "")
        ).strip()

        enabled_value = user.get(
            "enabled",
            "",
        )

        if not is_account_enabled(enabled_value):
            continue

        if (
            saved_username == username
            and saved_password == password
        ):
            if saved_condition not in ALLOWED_CONDITIONS:
                raise RuntimeError(
                    f"帳號 {saved_username} 的 condition 設定錯誤："
                    f"{saved_condition or '空白'}。"
                    "請設定為 structure、quality 或 hybrid。"
                )

            if not saved_room_id:
                raise RuntimeError(
                    f"帳號 {saved_username} 尚未設定 room_id。"
                    "請在 Users 工作表中填入聊天室編號，"
                    "例如 group01。"
                )

            return {
                "username": saved_username,
                "name": saved_name or saved_username,
                "condition": saved_condition,
                "room_id": saved_room_id,
            }

    return None
