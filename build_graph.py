#!/usr/bin/env python3
# build_graph.py
import os, sys, json, ast, re
from collections import defaultdict
from pathlib import Path

PY_IMPORT_RE = re.compile(r'^\s*import\s+([a-zA-Z0-9_\.]+)|^\s*from\s+([a-zA-Z0-9_\.]+)\s+import', re.M)

def is_code_file(p: Path) -> bool:
    return p.suffix in {'.py'}  # אפשר להרחיב לשפות נוספות

def module_name_from_path(root: Path, f: Path) -> str:
    rel = f.relative_to(root).with_suffix('')
    return '.'.join(rel.parts)

def parse_py_imports(text: str) -> set[str]:
    # עדיף AST על regex, אבל נוסיף גם regex כגיבוי לשורות לא סטנדרטיות
    out = set()
    try:
        tree = ast.parse(text)
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    out.add(a.name)
            elif isinstance(n, ast.ImportFrom):
                if n.module:
                    out.add(n.module)
    except SyntaxError:
        for m in PY_IMPORT_RE.finditer(text):
            g1, g2 = m.groups()
            if g1: out.add(g1)
            if g2: out.add(g2)
    return out

def belong_folder(root: Path, f: Path) -> str:
    rel = f.relative_to(root)
    if len(rel.parts) == 1:
        return "."  # שורש
    return str(rel.parent).replace(os.sep, '/')

def main():
    if len(sys.argv) < 2:
        print("usage: build_graph.py <repo_root>")
        sys.exit(1)

    root = Path(sys.argv[1]).resolve()
    files = [p for p in root.rglob('*') if p.is_file() and is_code_file(p)]

    # מיפוי: module_name -> קובץ
    mod2file = {}
    for f in files:
        mod = module_name_from_path(root, f)
        mod2file[mod] = f

    # צבירת תלויות בין קבצים (מודולים)
    deps = defaultdict(set)  # file -> set(file)
    for f in files:
        try:
            text = f.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        imports = parse_py_imports(text)
        src_mod = module_name_from_path(root, f)

        # ניסיון ל resolve: אם import הוא "a.b.c", ננסה התאמות הדרגתיות
        for imp in imports:
            cand = imp
            while cand:
                if cand in mod2file:
                    deps[src_mod].add(module_name_from_path(root, mod2file[cand]))
                    break
                # הסר רכיב אחרון
                if '.' in cand:
                    cand = cand.rsplit('.', 1)[0]
                else:
                    cand = None

    # בניית nodes
    folders = set()
    files_info = {}
    for f in files:
        folder = belong_folder(root, f)
        folders.add(folder)
        files_info[module_name_from_path(root,f)] = {
            "id": module_name_from_path(root,f),
            "type": "file",
            "path": str(f.relative_to(root)).replace(os.sep,'/'),
            "folder": folder
        }
    folder_nodes = [{"id": f"folder:{d}", "type": "folder", "name": d} for d in sorted(folders)]
    file_nodes = list(files_info.values())

    # בניית edges בין קבצים
    file_edges = []
    for s, tgts in deps.items():
        for t in tgts:
            if s in files_info and t in files_info:
                file_edges.append({
                    "source": s, "target": t, "type": "file_dep",
                    "cross_folder": files_info[s]["folder"] != files_info[t]["folder"]
                })

    # אגרגציה לקשרים בין תיקיות
    folder_pairs = defaultdict(int)
    for e in file_edges:
        sf = files_info[e["source"]]["folder"]
        tf = files_info[e["target"]]["folder"]
        if sf == tf: 
            continue
        key = tuple(sorted((sf, tf)))
        folder_pairs[key] += 1

    folder_edges = []
    for (a,b), w in folder_pairs.items():
        folder_edges.append({
            "source": f"folder:{a}", "target": f"folder:{b}",
            "type": "folder_dep", "weight": w
        })

    graph = {
        "nodes": folder_nodes + file_nodes,
        "edges": file_edges + folder_edges,
        "meta": {
            "root": str(root),
            "counts": {
                "folders": len(folder_nodes),
                "files": len(file_nodes),
                "file_edges": len(file_edges),
                "folder_edges": len(folder_edges),
            }
        }
    }
    out = root / "graph.json"
    out.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
