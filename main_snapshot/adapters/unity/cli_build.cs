// adapters/unity/cli_build.cs
using UnityEditor;
using System;
using System.IO;

public class IMU_CLI_Build {
    public static void BuildLinux64() {
        var scenes = new string[] {"Assets/Scene.unity"};
        var outPath = "Builds/Linux/IMUGame.x86_64";
        Directory.CreateDirectory("Builds/Linux");
        var report = BuildPipeline.BuildPlayer(scenes, outPath, BuildTarget.StandaloneLinux64, BuildOptions.None);
        if (report.summary.result != UnityEditor.Build.Reporting.BuildResult.Succeeded) {
            throw new Exception("unity_build_failed:" + report.summary.result.ToString());
        }
        Console.WriteLine("OK:" + outPath);
    }
}