import pandas as pd


def check_login(username, password):

    users = pd.read_csv(
        "database/users.csv"
    )

    user = users[
        (users["username"] == username) &
        (users["password"] == password)
    ]

    if len(user) == 1:
        return user.iloc[0]["name"]

    else:
        return None

