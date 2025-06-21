import pandas as pd


def seconds_to_hhmmss(seconds):
    if pd.isna(seconds):
        return ""
    seconds = int(seconds)
    hour = seconds // 3600
    minute = (seconds % 3600) // 60
    second = seconds % 60
    if hour >= 24:
        day = hour // 24
        hour = hour % 24
        return f"{day}d {hour:02d}:{minute:02d}:{second:02d}"
    return f"{hour:02d}:{minute:02d}:{second:02d}"
