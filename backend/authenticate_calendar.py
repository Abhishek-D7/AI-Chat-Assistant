import os
import sys
import shutil
from google_auth_oauthlib.flow import InstalledAppFlow

# Full Google Calendar access scope
SCOPES = ['https://www.googleapis.com/auth/calendar']

def authenticate_google_calendar():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cred_file = os.path.join(base_dir, "client_secret.json")
    token_file = os.path.join(base_dir, "token.json")
    root_token_file = os.path.join(os.path.dirname(base_dir), "token.json")

    if not os.path.exists(cred_file):
        print(f"❌ Error: {cred_file} not found.")
        sys.exit(1)

    print("=" * 60)
    print("🔐 GOOGLE CALENDAR AUTHENTICATION SETUP")
    print("=" * 60)
    print("1. Your browser will open with the Google Account consent screen.")
    print("2. Sign in with the Google Account whose calendar you want to use.")
    print("3. Click 'Allow' / 'Continue' to grant calendar permissions.")
    print("=" * 60)

    try:
        flow = InstalledAppFlow.from_client_secrets_file(cred_file, SCOPES)
        creds = flow.run_local_server(port=0, prompt="consent")
        
        # Save token to backend/token.json
        with open(token_file, "w") as token:
            token.write(creds.to_json())
        print(f"✅ Token successfully saved to: {token_file}")
        
        # Also copy to root directory for convenience
        try:
            shutil.copy2(token_file, root_token_file)
            print(f"✅ Token backup copied to: {root_token_file}")
        except Exception:
            pass

        print("=" * 60)
        print("🎉 SUCCESS! Google Calendar is now live.")
        print("Uvicorn will automatically reload and enable live calendar booking.")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False

if __name__ == "__main__":
    authenticate_google_calendar()
