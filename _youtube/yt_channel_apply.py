#!/usr/bin/env python3
"""Push the channel kit in channel.json to YouTube through the Data API v3.

Two-step, non-interactive OAuth (works from a headless box):

  python3 _youtube/yt_channel_apply.py auth-url
      -> prints a URL. Open it in any browser logged in as the channel owner,
         approve, and copy the FULL URL of the page you land on (it starts with
         http://localhost:8765/?... and will fail to load; that is expected).
  python3 _youtube/yt_channel_apply.py auth-code '<pasted localhost url>'
      -> stores the token in ~/.config/yt_channel/token.json

  python3 _youtube/yt_channel_apply.py show          # current channel state
  python3 _youtube/yt_channel_apply.py apply         # description, keywords,
                                                     # country, banner, playlists,
                                                     # a "Series" channel section
  python3 _youtube/yt_channel_apply.py apply --dry-run

What the API cannot do (do these in Studio by hand): channel name, handle,
profile picture, the channel trailer (needs an uploaded video first).

Requires: pip install google-api-python-client google-auth-oauthlib
Reuses the OAuth client of the gws CLI (~/.config/gws/client_secret.json),
project gen-lang-client-0205332528; the YouTube Data API is enabled on that
project on first run if the token carries the cloud-platform scope.
"""
import argparse, json, os, sys, pathlib

HERE = pathlib.Path(__file__).resolve().parent
CFG = HERE / "channel.json"
BANNER = HERE.parent / "assets" / "images" / "youtube" / "banner.png"
CLIENT_SECRET = pathlib.Path.home() / ".config" / "gws" / "client_secret.json"
TOKEN = pathlib.Path.home() / ".config" / "yt_channel" / "token.json"
REDIRECT = "http://localhost:8765/"
SCOPES = [
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/cloud-platform",
]


def flow():
    from google_auth_oauthlib.flow import Flow
    f = Flow.from_client_secrets_file(str(CLIENT_SECRET), scopes=SCOPES, redirect_uri=REDIRECT)
    return f


def cmd_auth_url(_):
    f = flow()
    url, state = f.authorization_url(access_type="offline", prompt="consent", include_granted_scopes="false")
    TOKEN.parent.mkdir(parents=True, exist_ok=True)
    # PKCE verifier must survive into the separate auth-code process
    (TOKEN.parent / "pkce").write_text(json.dumps({"state": state, "code_verifier": f.code_verifier}))
    print(url)


def cmd_auth_code(a):
    f = flow()
    saved = json.loads((TOKEN.parent / "pkce").read_text())
    f.code_verifier = saved["code_verifier"]
    f.fetch_token(authorization_response=a.url)
    TOKEN.parent.mkdir(parents=True, exist_ok=True)
    TOKEN.write_text(f.credentials.to_json())
    os.chmod(TOKEN, 0o600)
    print("token saved to", TOKEN)


def creds():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    if not TOKEN.exists():
        sys.exit("no token; run: auth-url, then auth-code '<url>'")
    c = Credentials.from_authorized_user_file(str(TOKEN), SCOPES)
    if c.expired and c.refresh_token:
        c.refresh(Request())
        TOKEN.write_text(c.to_json())
    return c


def enable_api(c):
    from google.auth.transport.requests import AuthorizedSession
    project = json.load(open(CLIENT_SECRET))["installed"]["project_id"]
    s = AuthorizedSession(c)
    base = f"https://serviceusage.googleapis.com/v1/projects/{project}/services/youtube.googleapis.com"
    r = s.get(base)
    if r.ok and r.json().get("state") == "ENABLED":
        return
    r = s.post(base + ":enable")
    print("enable youtube.googleapis.com:", r.status_code, r.text[:200])


def yt(c):
    from googleapiclient.discovery import build
    return build("youtube", "v3", credentials=c, cache_discovery=False)


def my_channel(y):
    r = y.channels().list(part="snippet,brandingSettings,contentDetails,statistics", mine=True).execute()
    items = r.get("items", [])
    if not items:
        sys.exit("no channel for this account")
    return items[0]


def cmd_show(_):
    c = creds(); enable_api(c); y = yt(c)
    ch = my_channel(y)
    print(json.dumps({
        "id": ch["id"], "title": ch["snippet"]["title"], "customUrl": ch["snippet"].get("customUrl"),
        "description": ch["snippet"].get("description"),
        "keywords": ch.get("brandingSettings", {}).get("channel", {}).get("keywords"),
        "country": ch.get("brandingSettings", {}).get("channel", {}).get("country"),
        "banner": ch.get("brandingSettings", {}).get("image", {}).get("bannerExternalUrl"),
        "stats": ch.get("statistics"),
    }, indent=2))
    pl = y.playlists().list(part="snippet", mine=True, maxResults=50).execute()
    for p in pl.get("items", []):
        print("playlist:", p["id"], p["snippet"]["title"])
    sec = y.channelSections().list(part="snippet,contentDetails", mine=True).execute()
    for s in sec.get("items", []):
        print("section:", s["id"], s["snippet"].get("type"), s["snippet"].get("title"), s.get("contentDetails"))


def quote_keywords(words):
    return " ".join(f'"{w}"' if " " in w else w for w in words)


def cmd_apply(a):
    cfg = json.load(open(CFG))
    c = creds(); enable_api(c); y = yt(c)
    ch = my_channel(y)
    cid = ch["id"]
    if cfg["channel_id"] and cid != cfg["channel_id"]:
        sys.exit(f"logged-in channel {cid} != channel.json {cfg['channel_id']}; refusing")
    dry = a.dry_run

    # 1. branding: description, keywords, country, language (+ banner url after upload)
    branding = ch.get("brandingSettings", {})
    channel_b = branding.setdefault("channel", {})
    channel_b["description"] = cfg["description"]
    channel_b["keywords"] = quote_keywords(cfg["keywords"])
    channel_b["country"] = cfg["country"]
    channel_b["defaultLanguage"] = cfg["default_language"]
    print(f"[branding] description {len(cfg['description'])} chars, {len(cfg['keywords'])} keywords, country {cfg['country']}")

    # 2. banner
    if BANNER.exists():
        print(f"[banner] upload {BANNER}")
        if not dry:
            from googleapiclient.http import MediaFileUpload
            media = MediaFileUpload(str(BANNER), mimetype="image/png")
            r = y.channelBanners().insert(media_body=media).execute()
            branding.setdefault("image", {})["bannerExternalUrl"] = r["url"]
            print("[banner] url", r["url"])
    else:
        print("[banner] missing", BANNER)

    if not dry:
        y.channels().update(part="brandingSettings", body={"id": cid, "brandingSettings": branding}).execute()
        print("[branding] updated")

    # 3. playlists (idempotent by title)
    existing = {}
    pl = y.playlists().list(part="snippet", mine=True, maxResults=50).execute()
    for p in pl.get("items", []):
        existing[p["snippet"]["title"]] = p["id"]
    ids = []
    for p in cfg["playlists"]:
        if p["title"] in existing:
            print("[playlist] exists:", p["title"]); ids.append(existing[p["title"]]); continue
        print("[playlist] create:", p["title"])
        if not dry:
            r = y.playlists().insert(part="snippet,status", body={
                "snippet": {"title": p["title"], "description": p["description"], "defaultLanguage": "en"},
                "status": {"privacyStatus": "public"},
            }).execute()
            ids.append(r["id"])

    # 4. one "Series" section listing all playlists
    secs = y.channelSections().list(part="snippet,contentDetails", mine=True).execute().get("items", [])
    for s in cfg["sections"]:
        if any(x["snippet"].get("type") == s["type"] and x["snippet"].get("title") == s["title"] for x in secs):
            print("[section] exists:", s["title"]); continue
        print("[section] create:", s["title"], "with", len(ids), "playlists")
        if not dry and ids:
            y.channelSections().insert(part="snippet,contentDetails", body={
                "snippet": {"type": s["type"], "title": s["title"], "position": 0},
                "contentDetails": {"playlists": ids},
            }).execute()
    print("done" + (" (dry run)" if dry else ""))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("auth-url").set_defaults(fn=cmd_auth_url)
    p = sub.add_parser("auth-code"); p.add_argument("url"); p.set_defaults(fn=cmd_auth_code)
    sub.add_parser("show").set_defaults(fn=cmd_show)
    p = sub.add_parser("apply"); p.add_argument("--dry-run", action="store_true"); p.set_defaults(fn=cmd_apply)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
