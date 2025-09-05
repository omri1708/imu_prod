# adapters/android/sign.py
# -*- coding: utf-8 -*-
import os, subprocess, shlex
from ..contracts import ResourceRequired

def sign_apk(apk_path: str, keystore_path: str, alias: str, storepass: str, keypass: str = None):
    """
    משתמש ב-apksigner (Android build-tools). אם חסר – ResourceRequired.
    """
    apksigner = os.environ.get("APK_SIGNER", "apksigner")
    # בדיקת קיום
    try:
        subprocess.run([apksigner, "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        raise ResourceRequired("Android build-tools (apksigner)",
                               "Install Android SDK build-tools, ensure 'apksigner' in PATH")

    cmd = f'{shlex.quote(apksigner)} sign --ks {shlex.quote(keystore_path)} --ks-key-alias {shlex.quote(alias)} ' \
          f'--ks-pass pass:{shlex.quote(storepass)} '
    if keypass:
        cmd += f'--key-pass pass:{shlex.quote(keypass)} '
    cmd += shlex.quote(apk_path)
    subprocess.run(cmd, shell=True, check=True)
    return {"ok": True, "apk": apk_path, "signed": True}