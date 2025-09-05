# broker/stream.py
# -*- coding: utf-8 -*-
import time, threading, queue, json
from dataclasses import dataclass
from typing import Dict, Optional, Iterator, Tuple, Any, List, Deque
import collections, os, math


# עדיפויות: ככל שהמספר גבוה יותר — חשוב יותר
PRIORITY = {"logic": 3, "telemetry": 2, "logs": 1}
DEFAULT_PRIORITY = "telemetry"

def _now() -> float:
    return time.perf_counter()

class TokenBucket:
    """דלי אסימונים קלאסי לשסתום זרימה (rps, burst)."""
    def __init__(self, rps: float, burst: int):
        self.rps = float(max(0.0, rps))
        self.burst = int(max(1, burst))
        self.tokens = float(self.burst)
        self.t_last = _now()
        self._lock = threading.Lock()

    def allow(self, cost: float = 1.0) -> bool:
        with self._lock:
            t = _now()
            dt = max(0.0, t - self.t_last)
            self.t_last = t
            self.tokens = min(self.burst, self.tokens + dt * self.rps)
            if self.tokens >= cost:
                self.tokens -= cost
                return True
            return False

class _Sub:
    def __init__(self, topic: str, max_queue: int, drop_notify=None):
        self.topic = topic
        self.max_queue = max(1, int(max_queue))
        self.q: Deque[Tuple[int, dict]] = collections.deque()  # (prio, event)
        self.dropped_total = 0
        self._lock = threading.Lock()
        self._pop_cv = threading.Condition(self._lock)
        self._drop_notify = drop_notify

    def push(self, prio: int, ev: dict):
        with self._lock:
            if len(self.q) < self.max_queue:
                self.q.append((prio, ev))
                self._pop_cv.notify_all()
                return True
            # תור מלא: אם העדיפות של החדש גבוהה מהנמוכה ביותר — החלף (drop-lowest)
            lowest_i = None
            lowest_p = 10**9
            for i, (p, _) in enumerate(self.q):
                if p < lowest_p:
                    lowest_p = p; lowest_i = i
            if lowest_i is not None and prio > lowest_p:
                # נשמור סטטיסטיקה על drop
                self.dropped_total += 1
                if self._drop_notify:
                    try: self._drop_notify(self.topic, "queue_full_replace")
                    except Exception: pass
                # זרוק את הנמוך
                self.q.rotate(-lowest_i)
                self.q.popleft()
                self.q.rotate(lowest_i)
                self.q.append((prio, ev))
                self._pop_cv.notify_all()
                return True
            # אחרת—נפיל את החדש
            self.dropped_total += 1
            if self._drop_notify:
                try: self._drop_notify(self.topic, "queue_full_drop_new")
                except Exception: pass
            return False

    def pop(self, timeout: float = 15.0) -> Optional[dict]:
        limit = _now() + timeout
        with self._lock:
            while not self.q:
                remaining = limit - _now()
                if remaining <= 0: return None
                self._pop_cv.wait(remaining)
            # קח את הגבוה ביותר
            best_i = 0; best_p = -1
            for i, (p, _) in enumerate(self.q):
                if p > best_p:
                    best_p = p; best_i = i
            self.q.rotate(-best_i)
            _, ev = self.q.popleft()
            self.q.rotate(best_i)
            return ev

class Broker:
    """
    ברוקר רב-נושאי:
    * Back-pressure גלובלי (טוקן-באקט + N*burst guard).
    * תורי מנוי בעדיפויות, החלפת נמוכים בגבוהים.
    * Throttling פר-נושא (rps/burst/max_queue).
    """

    def __init__(self):
        # קונפיג ברירת מחדל (ניתן לשינוי בזמן ריצה)
        self.global_bucket = TokenBucket(
            rps=float(os.environ.get("IMU_GLOBAL_RPS", "200.0")),
            burst=int(os.environ.get("IMU_GLOBAL_BURST", "2000"))
        )
        self.global_backlog_limit = int(os.environ.get("IMU_GLOBAL_BACKLOG", "50000"))  # N*burst guard
        self._topics: Dict[str, Dict[str, Any]] = {}
        self._subs: Dict[str, List[_Sub]] = {}
        self._lock = threading.RLock()
        self._metrics = {
            "published": 0,
            "rejected_global": 0,
            "rejected_topic": 0,
            "dropped_sub": 0,
            "by_topic": {}  # topic -> dict
        }

    def configure_topic(self, topic: str, *, rps: float = 100.0, burst: int = 1000, max_queue: int = 2000):
        with self._lock:
            self._topics[topic] = {
                "bucket": TokenBucket(rps, burst),
                "max_queue": int(max_queue)
            }
            self._metrics["by_topic"].setdefault(topic, {"pub":0,"rej":0,"subscribers":0})

    def subscribe(self, topic: str, *, max_queue: Optional[int] = None) -> _Sub:
        with self._lock:
            if topic not in self._topics:
                self.configure_topic(topic)  # ברירת מחדל
            tcfg = self._topics[topic]
            mq = max_queue if max_queue is not None else tcfg["max_queue"]
            sub = _Sub(topic, mq, drop_notify=self._on_sub_drop)
            self._subs.setdefault(topic, []).append(sub)
            self._metrics["by_topic"][topic]["subscribers"] = len(self._subs[topic])
            return sub

    def publish(self, topic: str, event: Dict[str, Any], *, priority: str = DEFAULT_PRIORITY) -> bool:
        prio = PRIORITY.get(priority, PRIORITY[DEFAULT_PRIORITY])
        # שסתום גלובלי
        if not self.global_bucket.allow():
            with self._lock:
                self._metrics["rejected_global"] += 1
            return False
        # N*burst guard ברמת backlog כולל
        if self._backlog_size() >= self.global_backlog_limit:
            with self._lock:
                self._metrics["rejected_global"] += 1
            return False
        # נושא
        with self._lock:
            if topic not in self._topics:
                self.configure_topic(topic)
            tcfg = self._topics[topic]
        if not tcfg["bucket"].allow():
            with self._lock:
                self._metrics["rejected_topic"] += 1
                self._metrics["by_topic"][topic]["rej"] += 1
            return False

        # הפצה לכל המנויים; אם אין מנויים—לא נצבור “רפאים”
        pushed_any = False
        with self._lock:
            subs = list(self._subs.get(topic, []))
        if not subs:
            with self._lock:
                self._metrics["published"] += 1
                self._metrics["by_topic"][topic]["pub"] += 1
            return True

        for sub in subs:
            ok = sub.push(prio, dict(event))
            if ok: pushed_any = True

        with self._lock:
            self._metrics["published"] += 1
            self._metrics["by_topic"][topic]["pub"] += 1
        return pushed_any

    def _backlog_size(self) -> int:
        total = 0
        with self._lock:
            for lst in self._subs.values():
                for s in lst:
                    total += len(s.q)
        return total

    def _on_sub_drop(self, topic: str, reason: str):
        with self._lock:
            self._metrics["dropped_sub"] += 1
            self._metrics["by_topic"].setdefault(topic, {"pub":0,"rej":0,"subscribers":0})

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._metrics))

    # סטרים ל-SSE (Server-Sent Events) עם heartbeat והשהיה אדיבה
    def sse_iter(self, sub: _Sub, *, heartbeat_sec: float = 10.0):
        last_hb = _now()
        while True:
            ev = sub.pop(timeout=heartbeat_sec)
            now = _now()
            if ev is None:
                # heartbeat
                yield f"event: hb\ndata: {{\"ts\": {time.time():.3f}}}\n\n".encode("utf-8")
                last_hb = now
                continue
            data = json.dumps(ev, ensure_ascii=False)
            yield f"event: msg\ndata: {data}\n\n".encode("utf-8")
            if now - last_hb > heartbeat_sec:
                yield f"event: hb\ndata: {{\"ts\": {time.time():.3f}}}\n\n".encode("utf-8")
                last_hb = now

# סינגלטון
broker = Broker()
# קונפיג נושאים רלוונטיים כברירת מחדל
for t, cfg in {
    "events":      dict(rps=200.0, burst=2000, max_queue=5000),
    "progress":    dict(rps=300.0, burst=3000, max_queue=8000),
    "timeline":    dict(rps=100.0, burst=1000, max_queue=4000),
    "logs":        dict(rps=500.0, burst=5000, max_queue=10000),
    "telemetry":   dict(rps=800.0, burst=8000, max_queue=12000),
}.items():
    broker.configure_topic(t, **cfg)

# -----old
@dataclass
class TopicPolicy:
    rps: float = 50.0        # max msgs/sec per-topic
    burst: int = 200         # burst tokens
    max_subscribers: int = 200
    priority_weights: Tuple[int,int,int] = (4, 2, 1)  # hi, normal, low


class Subscription:
    def __init__(self, topic: str, q: "queue.Queue[Tuple[int,dict]]"):
        self.topic = topic
        self.q = q


class StreamBroker:
    """
    ברוקר פר־נושא עם back-pressure גלובלי + תיעדוף: 0=high,1=normal,2=low
    """
    def __init__(self, global_capacity: int = 10000):
        self._topics: Dict[str, Dict[str, Any]] = {}
        self._tp_policy: Dict[str, TopicPolicy] = {}
        self._global_lock = threading.RLock()
        self._global_inflight = 0
        self._global_capacity = global_capacity

    def ensure_topic(self, topic: str, policy: Optional[TopicPolicy] = None):
        with self._global_lock:
            if topic in self._topics:
                return
            pol = policy or TopicPolicy()
            self._tp_policy[topic] = pol
            self._topics[topic] = {
                "bucket": TokenBucket(pol.rps, pol.burst),
                "subs": set(),  # of Subscription
            }

    def subscribe(self, topic: str, *, max_queue: int = 1000) -> Subscription:
        self.ensure_topic(topic)
        pol = self._tp_policy[topic]
        with self._global_lock:
            if len(self._topics[topic]["subs"]) >= pol.max_subscribers:
                raise RuntimeError("too_many_subscribers")
            q: "queue.Queue[Tuple[int,dict]]" = queue.Queue(maxsize=max_queue)
            sub = Subscription(topic, q)
            self._topics[topic]["subs"].add(sub)
            return sub

    def unsubscribe(self, sub: Subscription):
        with self._global_lock:
            self._topics.get(sub.topic, {}).get("subs", set()).discard(sub)

    def publish(self, topic: str, msg: dict, *, priority: int = 1) -> bool:
        """
        מחזיר True אם פורסם לכל המנויים, False אם נזרק עקב Back-pressure/Throttling.
        """
        self.ensure_topic(topic)
        priority = max(0, min(2, priority))
        bucket: TokenBucket = self._topics[topic]["bucket"]

        with self._global_lock:
            if self._global_inflight >= self._global_capacity:
                return False  # back-pressure גלובלי
            if not bucket.take(1):
                return False  # Throttling per-topic

            subs = list(self._topics[topic]["subs"])
            delivered = True
            for sub in subs:
                try:
                    sub.q.put_nowait((priority, msg))
                    self._global_inflight += 1
                except queue.Full:
                    delivered = False  # subscriber איטי – משליכים לפי back-pressure
            return delivered

    def drain(self, sub: Subscription, *, block: bool = True, timeout: float = 15.0) -> Optional[dict]:
        """
        שולף הודעה אחת מה־Queue של המנוי לפי תיעדוף.
        """
        deadline = time.time() + timeout
        while True:
            remaining = max(0.0, deadline - time.time())
            if remaining == 0 and not block:
                return None
            try:
                prio, msg = sub.q.get(timeout=min(1.0, remaining))
                with self._global_lock:
                    self._global_inflight = max(0, self._global_inflight - 1)
                return msg
            except queue.Empty:
                if not block:
                    return None
                if time.time() >= deadline:
                    return None

    def sse_iter(self, sub: Subscription) -> Iterator[bytes]:
        """
        מחזיר גנרטור של שורות SSE (bytes) עבור מנוי נתון.
        """
        try:
            while True:
                m = self.drain(sub, block=True, timeout=30.0)
                if m is None:
                    # keep-alive
                    yield b": ping\n\n"
                    continue
                data = json.dumps(m, ensure_ascii=False).encode("utf-8")
                yield b"event: msg\n"
                yield b"data: " + data + b"\n\n"
        finally:
            self.unsubscribe(sub)

# ברוקר גלובלי לשימוש השרת/פייפליין
_broker = StreamBroker(global_capacity=50000)