# tests/test_adapters_dry_run.py
import unittest
from engine.policy import AskAndProceedPolicy, UserSubspace, RequestContext
from engine.adapters_runner import AdaptersService
import asyncio
import pytest
from server.pipeline.run_adapter import run_adapter, DryRunError
from server.events.bus import EventBus
from server.security.provenance import ProvenanceStore
from server.state.ttl import TTLRules
from server.policy.enforcement import CapabilityPolicy

@pytest.mark.asyncio
async def test_android_dry():
    plan = await run_adapter({
        "adapter":"android","action":"build",
        "args":{"project_dir":"/proj/android"},
        "require":["android-sdk"]
    }, dry=True, event_bus=EventBus(), prov=ProvenanceStore(base_dir=__import__("pathlib").Path("./.prov_test")), ttl=TTLRules(), policy=CapabilityPolicy())
    assert "cmd" in plan and plan["cmd"][0] in ("gradle","./gradlew")

@pytest.mark.asyncio
async def test_ios_dry():
    plan = await run_adapter({"adapter":"ios","action":"build","args":{"workspace":"App.xcworkspace","scheme":"App"}}, dry=True,
                             event_bus=EventBus(), prov=ProvenanceStore(base_dir=__import__("pathlib").Path("./.prov_test")), ttl=TTLRules(), policy=CapabilityPolicy())
    assert plan["cmd"][0] == "xcodebuild"

@pytest.mark.asyncio
async def test_unity_dry():
    plan = await run_adapter({"adapter":"unity","action":"build","args":{"project_dir":"/proj/unity","target":"Android"}}, dry=True,
                             event_bus=EventBus(), prov=ProvenanceStore(base_dir=__import__("pathlib").Path("./.prov_test")), ttl=TTLRules(), policy=CapabilityPolicy())
    assert plan["cmd"][0] in ("unity","Unity")

@pytest.mark.asyncio
async def test_cuda_dry():
    plan = await run_adapter({"adapter":"cuda","action":"build","args":{"file":"kernel.cu","out":"kernel.out"}}, dry=True,
                             event_bus=EventBus(), prov=ProvenanceStore(base_dir=__import__("pathlib").Path("./.prov_test")), ttl=TTLRules(), policy=CapabilityPolicy())
    assert plan["cmd"][0] == "nvcc"

@pytest.mark.asyncio
async def test_k8s_dry():
    plan = await run_adapter({"adapter":"k8s","action":"deploy","args":{"manifest":"k8s/app.yaml"}}, dry=True,
                             event_bus=EventBus(), prov=ProvenanceStore(base_dir=__import__("pathlib").Path("./.prov_test")), ttl=TTLRules(), policy=CapabilityPolicy())
    assert plan["cmd"][0] == "kubectl"


class T(unittest.TestCase):
    def setUp(self):
        from streaming.broker import StreamBroker
        self.policy = AskAndProceedPolicy({})
        self.broker = StreamBroker()
        self.svc = AdaptersService(self.policy, self.broker)
        self.ctx = RequestContext(UserSubspace("u",3,3600,False,False,True), "test")

    def test_android(self):
        p = self.svc.dry_run("android", {"module":"app","variant":"Release"}, self.ctx)
        self.assertIn("gradlew", p.commands[0])

    def test_ios(self):
        p = self.svc.dry_run("ios", {"scheme":"Foo","configuration":"Release"}, self.ctx)
        self.assertIn("xcodebuild", p.commands[0])

    def test_unity(self):
        p = self.svc.dry_run("unity", {"projectPath":"Proj","buildTarget":"WebGL"}, self.ctx)
        self.assertIn("unity -quit -batchmode", p.commands[0])

    def test_cuda(self):
        p = self.svc.dry_run("cuda", {"src":"k.cu","arch":"sm_90"}, self.ctx)
        self.assertIn("nvcc", p.commands[0])

    def test_k8s(self):
        p = self.svc.dry_run("k8s", {"manifest":"d.yaml","namespace":"ns"}, self.ctx)
        self.assertIn("kubectl apply", p.commands[0])

if __name__ == "__main__":
    unittest.main()