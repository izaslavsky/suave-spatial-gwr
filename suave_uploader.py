# suave_uploader.py

import io
import requests

def upload_to_suave(df, survey_name, user, password, referer, dzc_file=None):
    """
    Upload a DataFrame to SuAVE as a new survey.
    Returns: (success_flag, message, new_url)
    """
    try:
        # Step 1: Log in with session
        s = requests.Session()
        headers = {
            "User-Agent": "suave user agent",
            "referer": referer
        }

        login_response = s.post(
            referer,
            headers=headers,
            data={
                "user": user,
                "pass": password,
                "remember-me": "true"
            }
        )

        if login_response.status_code != 200:
            return (False, f"Login failed (status {login_response.status_code})", None)

        # Step 2: Prepare the CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        files = {
            "file": (f"{survey_name}.csv", csv_buffer.getvalue())
        }

        data = {
            "name": survey_name,
            "user": user
        }

        if dzc_file:
            data["dzc"] = dzc_file

        upload_url = referer + "uploadCSV"
        upload_response = s.post(upload_url, files=files, data=data, headers=headers)

        if upload_response.status_code == 200:
            new_survey_url = f"{referer}main/file={user}_{survey_name}.csv"
            return (True, "✅ Survey uploaded successfully!", new_survey_url)
        else:
            return (False, f"Upload failed ({upload_response.status_code}) – {upload_response.text}", None)

    except Exception as e:
        return (False, f"Upload failed due to error: {e}", None)
