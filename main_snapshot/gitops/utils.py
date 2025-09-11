# gitops/utils.py
from __future__ import annotations
import subprocess, shutil, os, platform
from typing import List, Tuple, Optional

class GitError(RuntimeError): ...

def have_git() -> bool:
    return shutil.which("git") is not None

def install_hint_git() -> str:
    sys = platform.system().lower()
    if "windows" in sys: return "winget install -e --id Git.Git"
    if "darwin"  in sys: return "brew install git"
    return "sudo apt-get update && sudo apt-get install -y git"

def run_git(args: List[str], cwd: str) -> Tuple[int, str]:
    try:
        p = subprocess.run(["git"]+args, cwd=cwd, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        return p.returncode, p.stdout
    except Exception as e:
        raise GitError(f"git {' '.join(args)} failed: {e}")

def ensure_repo(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    if not os.path.isdir(os.path.join(path, ".git")):
        rc, out = run_git(["init","-b","main"], cwd=path)
        if rc != 0: raise GitError(out)
    return path

def add_all(path: str, patterns: Optional[List[str]] = None) -> str:
    pats = patterns or ["."]
    rc, out = run_git(["add"]+pats, cwd=path)
    if rc != 0: raise GitError(out)
    return out

def commit(path: str, message: str, author: Optional[str] = None) -> str:
    env = os.environ.copy()
    if author:
        env["GIT_AUTHOR_NAME"]=author
        env["GIT_AUTHOR_EMAIL"]=f"{author}@imu.local"
        env["GIT_COMMITTER_NAME"]=author
        env["GIT_COMMITTER_EMAIL"]=f"{author}@imu.local"
    p = subprocess.run(["git","commit","-m",message], cwd=path, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    if p.returncode != 0: raise GitError(p.stdout)
    return p.stdout

def current_branch(path: str) -> str:
    rc, out = run_git(["rev-parse","--abbrev-ref","HEAD"], cwd=path)
    if rc != 0: raise GitError(out)
    return out.strip()

def create_branch(path: str, name: str, base: str = "main") -> str:
    rc, out = run_git(["checkout","-B",name, base], cwd=path)
    if rc != 0: raise GitError(out)
    return name

def set_remote(path: str, name: str, url: str) -> str:
    rc, _ = run_git(["remote","remove",name], cwd=path)  # ignore failures
    rc, out = run_git(["remote","add",name,url], cwd=path)
    if rc != 0: raise GitError(out)
    return out

def push(path: str, remote: str, branch: str) -> str:
    rc, out = run_git(["push", remote, branch], cwd=path)
    if rc != 0: raise GitError(out)
    return out

def status(path: str) -> str:
    rc, out = run_git(["status","--porcelain","-b"], cwd=path)
    if rc != 0: raise GitError(out)
    return out

def merge(path: str, target: str, source: str, no_ff: bool = True) -> str:
    # checkout target then merge source
    rc, out = run_git(["checkout",target], cwd=path)
    if rc != 0: raise GitError(out)
    args = ["merge"]
    if no_ff: args.append("--no-ff")
    args.append(source)
    rc, out = run_git(args, cwd=path)
    if rc != 0: raise GitError(out)
    return out