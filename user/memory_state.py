# imu_repo/user/memory_state.py
from __future__ import annotations
import os, json, time, hashlib
from typing import Dict, Any, List, Optional

from .semvec import embed, cosine


class CryptoStream:
    """
    הצפנה במנוחה: XOR עם keystream על בסיס SHA256(key||counter).
    זה לא AES; לא טוענים ל-HSM. מספק "at rest" כמתבקש ללא תלות חיצונית.
    """
    def __init__(self, key: bytes):
        self.key = key
    def _block(self, n: int) -> bytes:
        h = hashlib.sha256(self.key + n.to_bytes(8,"big")).digest()
        return h
    def crypt(self, data: bytes) -> bytes:
        out=bytearray()
        for i,b in enumerate(data):
            blk=self._block(i//32)
            out.append(b ^ blk[i%32])
        return bytes(out)


def _now() -> float: return time.time()


class MemoryItem:
    def __init__(self, kind: str, text: str, meta: Dict[str,Any], conf: float, ttl_s: int):
        self.kind=kind; self.text=text; self.meta=meta
        self.conf=float(conf); self.ttl_s=int(ttl_s)
        self.created=_now(); self.updated=self.created
        self.vec = embed(text)

    def to_dict(self) -> Dict[str,Any]:
        return {"kind":self.kind,"text":self.text,"meta":self.meta,"conf":self.conf,"ttl_s":self.ttl_s,"created":self.created,"updated":self.updated,"vec":self.vec}

    @staticmethod
    def from_dict(d: Dict[str,Any]) -> "MemoryItem":
        m=MemoryItem(d["kind"], d["text"], d.get("meta",{}), d.get("conf",0.5), d.get("ttl_s", 90*24*3600))
        m.created=d.get("created",_now()); m.updated=d.get("updated",m.created); m.vec=d.get("vec") or embed(m.text)
        return m


class MemoryState:
    """
    T0: זיכרון זמני/קונטקסט (בתוך-מפגש) — dictionary in-ram
    T1: זיכרון קצר — קובץ json פר session
    T2: זיכרון ארוך — קובץ מוצפן פר user
    """
    def __init__(self, user_id: str, root: str = ".imu_state/mem", master_key_path: str = ".imu_state/mem.key"):
        self.user_id=user_id
        self.root=root
        os.makedirs(root, exist_ok=True)
        # master key
        if not os.path.exists(master_key_path):
            with open(master_key_path,"wb") as f: f.write(os.urandom(32))
        with open(master_key_path,"rb") as f: self.master = f.read()
        self.cs = CryptoStream(self.master)

        self.t0: Dict[str,Any] = {}
        self.t1_path = os.path.join(root, f"{user_id}.t1.json")
        self.t2_path = os.path.join(root, f"{user_id}.t2.enc")
        self._ensure_files()

    def _ensure_files(self):
        if not os.path.exists(self.t1_path):
            with open(self.t1_path,"w",encoding="utf-8") as f: json.dump({"items":[]}, f)
        if not os.path.exists(self.t2_path):
            with open(self.t2_path,"wb") as f: f.write(self.cs.crypt(json.dumps({"items":[]}).encode()))

    # ---- T1 ----
    def _load_t1(self) -> List[MemoryItem]:
        with open(self.t1_path,"r",encoding="utf-8") as f: obj=json.load(f)
        return [MemoryItem.from_dict(x) for x in obj.get("items",[])]

    def _save_t1(self, items: List[MemoryItem]):
        with open(self.t1_path,"w",encoding="utf-8") as f:
            json.dump({"items":[m.to_dict() for m in items]}, f, ensure_ascii=False, indent=2)

    # ---- T2 (encrypted) ----
    def _load_t2(self) -> List[MemoryItem]:
        with open(self.t2_path,"rb") as f:
            data = self.cs.crypt(f.read())
        obj = json.loads(data.decode())
        return [MemoryItem.from_dict(x) for x in obj.get("items",[])]

    def _save_t2(self, items: List[MemoryItem]):
        data=json.dumps({"items":[m.to_dict() for m in items]}, ensure_ascii=False).encode()
        with open(self.t2_path,"wb") as f: f.write(self.cs.crypt(data))

    # ---- API ----
    def t0_put(self, k: str, v: Any): self.t0[k]=v
    def t0_get(self, k: str, dv: Any=None): return self.t0.get(k,dv)
    def t0_clear(self): self.t0.clear()

    def add_observation(self, kind: str, text: str, meta: Dict[str,Any], conf: float=0.6, ttl_s: int=90*24*3600, tier: str="T1"):
        m = MemoryItem(kind, text, meta, conf, ttl_s)
        if tier=="T1":
            items=self._load_t1(); items.append(m); self._save_t1(items)
        else:
            items=self._load_t2(); items.append(m); self._save_t2(items)

    def query(self, text: str, k: int = 5) -> List[Dict[str,Any]]:
        q=embed(text)
        res=[]
        for m in self._load_t2() + self._load_t1():
            if _expired(m): continue
            res.append((cosine(q,m.vec), m))
        res.sort(key=lambda x:x[0], reverse=True)
        return [{"score":s, **x.to_dict()} for s,x in res[:k]]

    def consolidate(self, salience_thresh: float = 0.75, min_conf: float = 0.55):
        """
        T1→T2: אם פריט רלוונטי/בעל־משמעות (salience לפי וקטור לשאלות אחרונות) — מקדם ל-T2.
        כאן מיישמים כלל פשוט: conf>=min_conf ו/או בוצע עליו MATCH לאחרונה (נשמר ב-T0).
        """
        t1=self._load_t1(); t2=self._load_t2()
        keep=[]; moved=0
        recent = self.t0.get("__recent_queries__", [])
        rec_vec = embed(" ".join(recent)) if recent else None
        for m in t1:
            if _expired(m): 
                continue
            sal = cosine(rec_vec,m.vec) if rec_vec else 0.0
            if (m.conf>=min_conf) or (sal>=salience_thresh):
                # conflict-resolution: אם יש בטקסט דומה ב-T2 — מאחדים על־פי recency+conf
                merged=False
                for i,mm in enumerate(t2):
                    if cosine(m.vec, mm.vec) >= 0.92:
                        # נעדיף חדש אם conf גבוה יותר או חדש יותר
                        pick = m if (m.conf>mm.conf or m.updated>mm.updated) else mm
                        t2[i] = pick; merged=True; break
                if not merged:
                    t2.append(m)
                moved+=1
            else:
                keep.append(m)
        self._save_t1(keep)
        self._save_t2(t2)
        return {"moved": moved, "remaining": len(keep), "t2": len(t2)}

    def decay(self):
        # מוחק פריטים שפג תוקפם
        t1=[m for m in self._load_t1() if not _expired(m)]
        t2=[m for m in self._load_t2() if not _expired(m)]
        self._save_t1(t1); self._save_t2(t2)

    def erase(self, predicate: Dict[str,Any]):
        """
        מחיקה עפ"י מדיניות (למשל {"kind":"preference"} או {"meta":{"key":"value"}})
        """
        def match(m: MemoryItem) -> bool:
            if "kind" in predicate and m.kind != predicate["kind"]: return False
            if "text" in predicate and predicate["text"] not in m.text: return False
            if "meta" in predicate:
                for k,v in predicate["meta"].items():
                    if m.meta.get(k)!=v: return False
            return True
        t1=[m for m in self._load_t1() if not match(m)]
        t2=[m for m in self._load_t2() if not match(m)]
        self._save_t1(t1); self._save_t2(t2)


def _expired(m: MemoryItem) -> bool:
    return (m.ttl_s>0) and ((_now() - m.updated) > m.ttl_s)