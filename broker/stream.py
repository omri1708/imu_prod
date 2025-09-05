# broker/stream.py
# -*- coding: utf-8 -*-
import time, threading, queue, json
from dataclasses import dataclass
from typing import Dict, Optional, Iterator, Tuple, Any, List, Deque
import collections, os, math
import time, threading, collections, json, os, math, random
from typing import Deque, Dict, Any, Optional, Tuple, List
from .policy import DropPolicy, load_hint, WFQ

PRIORITY = {"logic": 3, "telemetry": 2, "logs": 1}
DEFAULT_PRIORITY = "telemetry"

def _now() -> float: return time.perf_counter()

class TokenBucket:
    def __init__(self, rps: float, burst: int):
        self.rps = float(max(0.0, rps))
        self.burst = int(max(1, burst))
        self.tokens = float(self.burst)
        self.t_last = _now()
        self._lock = threading.Lock()

    def allow(self, cost: float = 1.0) -> bool:
        with self._lock:
            t = _now(); dt = max(0.0, t - self.t_last); self.t_last = t
            self.tokens = min(self.burst, self.tokens + dt * self.rps)
            if self.tokens >= cost:
                self.tokens -= cost; return True
            return False

class _Sub:
    def __init__(self, topic: str, max_queue: int, drop_policy: str, drop_notify=None):
        self.topic = topic
        self.max_queue = max(1, int(max_queue))
        self.policy = drop_policy
        self.q: Deque[Tuple[int, dict]] = collections.deque()
        self.dropped_total = 0
        self._lock = threading.Lock()
        self._pop_cv = threading.Condition(self._lock)
        self._drop_notify = drop_notify

    def _drop(self, reason: str):
        self.dropped_total += 1
        if self._drop_notify:
            try: self._drop_notify(self.topic, reason)
            except Exception: pass

    def push(self, prio: int, ev: dict) -> bool:
        with self._lock:
            if len(self.q) < self.max_queue:
                self.q.append((prio, ev)); self._pop_cv.notify_all(); return True

            pol = self.policy
            if pol == DropPolicy.TAIL_DROP:
                self._drop("queue_full_tail_drop"); return False
            if pol == DropPolicy.HEAD_DROP:
                if self.q:
                    self.q.popleft(); self._drop("queue_full_head_drop")
                    self.q.append((prio, ev)); self._pop_cv.notify_all(); return True
                self._drop("queue_full_head_drop_empty"); return False
            if pol == DropPolicy.LOWEST_PRIORITY_REPLACE:
                lowest_i, lowest_p = None, 10**9
                for i, (p, _) in enumerate(self.q):
                    if p < lowest_p: lowest_p, lowest_i = p, i
                if lowest_i is not None and prio > lowest_p:
                    self._drop("queue_full_replace_lowest")
                    self.q.rotate(-lowest_i); self.q.popleft(); self.q.rotate(lowest_i)
                    self.q.append((prio, ev)); self._pop_cv.notify_all(); return True
                self._drop("queue_full_keep"); return False
            if pol == DropPolicy.RANDOM_EARLY_DROP:
                # הסתברות לזריקה עולה עם עומס
                prob = min(0.9, len(self.q)/float(self.max_queue))
                if random.random() < prob:
                    self._drop("queue_red_drop"); return False
                self.q.append((prio, ev)); self._pop_cv.notify_all(); return True

            # ברירת מחדל שמרנית
            self._drop("queue_full_default_drop"); return False

    def pop(self, timeout: float = 15.0) -> Optional[dict]:
        deadline = _now() + timeout
        with self._lock:
            while not self.q:
                left = deadline - _now()
                if left <= 0: return None
                self._pop_cv.wait(left)
            best_i, best_p = 0, -1
            for i, (p, _) in enumerate(self.q):
                if p > best_p: best_p, best_i = p, i
            self.q.rotate(-best_i); _, ev = self.q.popleft(); self.q.rotate(best_i)
            return ev

class Broker:
    """
    * Back-pressure גלובלי (דלי אסימונים + שמירת backlog כולל).
    * Throttling פר-נושא.
    * WFQ בין נושאים (חלוקה הוגנת לפי משקל, ע"י vtime).
    * מדיניות זריקה לתורי המנויים.
    """
    def __init__(self):
        self.global_bucket = TokenBucket(float(os.environ.get("IMU_GLOBAL_RPS", "400.0")),
                                         int(os.environ.get("IMU_GLOBAL_BURST", "4000")))
        self.global_backlog_soft = int(os.environ.get("IMU_GLOBAL_BACKLOG_SOFT", "50000"))
        self.global_backlog_hard = int(os.environ.get("IMU_GLOBAL_BACKLOG_HARD", "80000"))
        self._topics: Dict[str, Dict[str, Any]] = {}
        self._subs: Dict[str, List[_Sub]] = {}
        self._lock = threading.RLock()
        self._metrics = {"published":0,"rejected_global":0,"rejected_topic":0,"dropped_sub":0,"by_topic":{}}
        self._wfq = WFQ()

    def configure_topic(self, topic: str, *, rps: float = 150.0, burst: int = 1500,
                        max_queue: int = 3000, drop_policy: str = DropPolicy.LOWEST_PRIORITY_REPLACE, weight: float = 1.0):
        with self._lock:
            self._topics[topic] = {"bucket": TokenBucket(rps, burst),
                                   "max_queue": int(max_queue),
                                   "drop_policy": drop_policy,
                                   "weight": float(max(0.1, weight)),
                                   "vstart": 0.0}
            self._metrics["by_topic"].setdefault(topic, {"pub":0,"rej":0,"subscribers":0})

    def subscribe(self, topic: str, *, max_queue: Optional[int] = None, drop_policy: Optional[str]=None) -> _Sub:
        with self._lock:
            if topic not in self._topics: self.configure_topic(topic)
            tcfg = self._topics[topic]
            mq = max_queue if max_queue is not None else tcfg["max_queue"]
            pol = drop_policy if drop_policy is not None else tcfg["drop_policy"]
            sub = _Sub(topic, mq, pol, drop_notify=self._on_sub_drop)
            self._subs.setdefault(topic, []).append(sub)
            self._metrics["by_topic"][topic]["subscribers"] = len(self._subs[topic])
            return sub

    def publish(self, topic: str, event: Dict[str, Any], *, priority: str = DEFAULT_PRIORITY) -> bool:
        prio = PRIORITY.get(priority, PRIORITY[DEFAULT_PRIORITY])

        # שסתום גלובלי + N*burst guard
        if not self.global_bucket.allow():
            with self._lock: self._metrics["rejected_global"] += 1
            return False
        backlog = self._backlog_size()
        hint = load_hint(backlog, self.global_backlog_soft, self.global_backlog_hard)
        if hint == "critical":
            with self._lock: self._metrics["rejected_global"] += 1
            return False

        with self._lock:
            if topic not in self._topics: self.configure_topic(topic)
            tcfg = self._topics[topic]
            # WFQ vstart (לפי משקל)
            active_sum = sum(t["weight"] for t in self._topics.values())
            vtime = self._wfq.tick(active_sum)
            tcfg["vstart"] = vtime

        if not tcfg["bucket"].allow():
            with self._lock:
                self._metrics["rejected_topic"] += 1
                self._metrics["by_topic"][topic]["rej"] += 1
            return False

        with self._lock:
            subs = list(self._subs.get(topic, []))
        if not subs:
            with self._lock:
                self._metrics["published"] += 1
                self._metrics["by_topic"][topic]["pub"] += 1
            return True

        pushed_any = False
        for s in subs:
            ok = s.push(prio, dict(event))
            pushed_any |= ok
        with self._lock:
            self._metrics["published"] += 1
            self._metrics["by_topic"][topic]["pub"] += 1
        return pushed_any

    def _backlog_size(self) -> int:
        total = 0
        with self._lock:
            for subs in self._subs.values():
                for s in subs: total += len(s.q)
        return total

    def _on_sub_drop(self, topic: str, reason: str):
        with self._lock:
            self._metrics["dropped_sub"] += 1
            self._metrics["by_topic"].setdefault(topic, {"pub":0,"rej":0,"subscribers":0})

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._metrics))

# סינגלטון וקונפיג בסיסי
broker = Broker()
for t, cfg in {
    "events":    dict(rps=250.0, burst=2500, max_queue=6000, weight=1.0),
    "progress":  dict(rps=400.0, burst=4000, max_queue=8000, weight=1.5),
    "timeline":  dict(rps=150.0, burst=1500, max_queue=5000, weight=1.0),
    "logs":      dict(rps=600.0, burst=6000, max_queue=12000, weight=0.7),
    "telemetry": dict(rps=900.0, burst=9000, max_queue=14000, weight=1.2),
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