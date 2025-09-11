# imu_repo/ui/mobile.py
from __future__ import annotations
import os, shutil, subprocess
from typing import Optional

class ResourceRequired(Exception): ...

ANDROID_MAIN = """package com.imu.app;

import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {
  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    TextView tv = new TextView(this);
    tv.setText("Hello IMU!");
    setContentView(tv);
  }
}
"""

BUILD_GRADLE_PRJ = """buildscript {
  repositories { google(); mavenCentral() }
  dependencies { classpath 'com.android.tools.build:gradle:8.1.0' }
}
allprojects { repositories { google(); mavenCentral() } }
"""

SETTINGS_GRADLE = "include ':app'\nrootProject.name = 'IMUApp'\n"

BUILD_GRADLE_APP = """apply plugin: 'com.android.application'

android {
  namespace "com.imu.app"
  compileSdkVersion 34
  defaultConfig {
    applicationId "com.imu.app"
    minSdkVersion 24
    targetSdkVersion 34
    versionCode 1
    versionName "1.0"
  }
  buildTypes {
    release { minifyEnabled false }
  }
}

dependencies {
  implementation 'androidx.appcompat:appcompat:1.6.1'
}
"""

MANIFEST_XML = """<?xml version="1.0" encoding="utf-8"?>
<manifest package="com.imu.app" xmlns:android="http://schemas.android.com/apk/res/android">
  <application android:label="IMUApp" android:allowBackup="true">
    <activity android:name=".MainActivity">
      <intent-filter>
        <action android:name="android.intent.action.MAIN"/>
        <category android:name="android.intent.category.LAUNCHER"/>
      </intent-filter>
    </activity>
  </application>
</manifest>
"""

def ensure_android_project(path: str):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "build.gradle"), "w") as f: f.write(BUILD_GRADLE_PRJ)
    with open(os.path.join(path, "settings.gradle"), "w") as f: f.write(SETTINGS_GRADLE)
    app = os.path.join(path, "app")
    main_java = os.path.join(app, "src","main","java","com","imu","app")
    main_res  = os.path.join(app, "src","main","res")
    os.makedirs(main_java, exist_ok=True)
    os.makedirs(main_res, exist_ok=True)
    with open(os.path.join(app, "build.gradle"), "w") as f: f.write(BUILD_GRADLE_APP)
    with open(os.path.join(app, "src","main","AndroidManifest.xml"), "w") as f: f.write(MANIFEST_XML)
    with open(os.path.join(main_java, "MainActivity.java"), "w") as f: f.write(ANDROID_MAIN)

def require_env(var: str):
    val = os.environ.get(var)
    if not val:
        raise ResourceRequired(f"{var} environment variable required")
    return val

def build_debug(path: str) -> str:
    # Requires ANDROID_SDK_ROOT and 'gradle' on PATH
    require_env("ANDROID_SDK_ROOT")
    try:
        subprocess.run(["gradle","-v"], check=True, capture_output=True)
    except Exception as e:
        raise ResourceRequired("Gradle required on PATH") from e
    subprocess.run(["gradle","assembleDebug"], cwd=path, check=True)
    apk = os.path.join(path, "app","build","outputs","apk","debug","app-debug.apk")
    if not os.path.exists(apk):
        raise RuntimeError("APK not found after build")
    return apk
