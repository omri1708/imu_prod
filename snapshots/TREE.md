```
.
├── adapters
│   ├── __init__.py
│   ├── adapter_runner.py
│   ├── andriod.py
│   ├── android
│   │   ├── build.py
│   │   └── sign.py
│   ├── android_build.py
│   ├── android_builder.py
│   ├── async_tasks.py
│   ├── base.py
│   ├── build_android.py
│   ├── build_ios.py
│   ├── contracts
│   │   ├── android_build.json
│   │   ├── android_gradle.json
│   │   ├── base.py
│   │   ├── cuda_job.json
│   │   ├── cuda_nvcc.json
│   │   ├── ios_build.json
│   │   ├── ios_xcode.json
│   │   ├── k8s_apply.json
│   │   ├── k8s_deploy.json
│   │   ├── unity_build.json
│   │   └── unity_cli.json
│   ├── contracts.py
│   ├── cuda
│   │   ├── job_runner.py
│   │   └── runner.py
│   ├── cuda_jobs.py
│   ├── cuda_runner.py
│   ├── cuda.py
│   ├── db
│   │   └── sqlite_sandbox.py
│   ├── db_localqueue.py
│   ├── docker
│   │   ├── build.py
│   │   └── run.py
│   ├── docker_sign.py
│   ├── fs_sandbox.py
│   ├── generated
│   │   ├── db-pg-backup
│   │   │   ├── cli_templates.json
│   │   │   ├── contract.json
│   │   │   └── README.md
│   │   ├── db-pg-restore
│   │   │   ├── cli_templates.json
│   │   │   └── contract.json
│   │   ├── gpu-cuda-smi-log
│   │   │   ├── cli_templates.json
│   │   │   └── contract.json
│   │   ├── gpu-nvml-metrics
│   │   │   ├── cli_templates.json
│   │   │   └── contract.json
│   │   ├── helm-rollback
│   │   │   ├── cli_templates.json
│   │   │   └── contract.json
│   │   ├── helm-template
│   │   │   ├── cli_templates.json
│   │   │   └── contract.json
│   │   ├── helm-upgrade
│   │   │   ├── cli_templates.json
│   │   │   └── contract.json
│   │   ├── infra-ansible-apply
│   │   │   ├── cli_templates.json
│   │   │   └── contract.json
│   │   ├── infra-ansible-galaxy
│   │   │   ├── cli_templates.json
│   │   │   └── contract.json
│   │   ├── infra-terraform-apply
│   │   │   ├── cli_templates.json
│   │   │   └── contract.json
│   │   ├── queue-kafka-produce
│   │   │   ├── cli_templates.json
│   │   │   └── contract.json
│   │   ├── queue-nats-publish
│   │   │   ├── cli_templates.json
│   │   │   └── contract.json
│   │   ├── queue-rabbitmq-publish
│   │   │   ├── cli_templates.json
│   │   │   └── contract.json
│   │   ├── queue-redis-subscribe
│   │   │   ├── cli_templates.json
│   │   │   ├── contract.json
│   │   │   └── README.md
│   │   ├── scm-git-clone
│   │   │   ├── cli_templates.json
│   │   │   └── contract.json
│   │   ├── storage-azure-blob-sync
│   │   │   ├── cli_templates.json
│   │   │   └── contract.json
│   │   ├── storage-gcs-sync
│   │   │   ├── cli_templates.json
│   │   │   └── contract.json
│   │   └── storage-s3-sync
│   │       ├── cli_templates.json
│   │       ├── contract.json
│   │       └── README.md
│   ├── gpu
│   │   ├── cuda_runner.py
│   │   └── pipeline.py
│   ├── http_fetch.py
│   ├── installer.py
│   ├── ios
│   │   ├── build.py
│   │   └── sign.py
│   ├── ios_build.py
│   ├── ios_builder.py
│   ├── ios.py
│   ├── k8s
│   │   ├── cuda_job.py
│   │   ├── deploy_plugin.py
│   │   ├── deploy.py
│   │   ├── k8s_deployer.py
│   │   └── rollout.py
│   ├── k8s_deploy.py
│   ├── k8s_plugin.py
│   ├── k8s_uploader.py
│   ├── k8s.py
│   ├── mappings.py
│   ├── mobile
│   │   ├── android_build.py
│   │   └── ios_build.py
│   ├── net_sandbox.py
│   ├── pack_a
│   │   └── manifest.py
│   ├── pkg_mappings.py
│   ├── provenance_store.py
│   ├── redis_resp.py
│   ├── registry.py
│   ├── synth
│   │   ├── generator.py
│   │   └── registry.py
│   ├── tests
│   │   └── test_adapters.py
│   ├── unity
│   │   ├── __init__.py
│   │   ├── build_unity.py
│   │   ├── build.py
│   │   ├── cli_build.cs
│   │   ├── cli_build.py
│   │   ├── cli.py
│   │   └── scenes.py
│   ├── unity_cli.py
│   ├── unity_cly.py
│   ├── unity.py
│   └── validate.py
├── alerts
│   ├── alerts.py
│   └── notifier.py
├── api
│   ├── run_adapter_http.py
│   └── stream_http.py
├── argocd
│   ├── apps
│   │   ├── app-of-apps.yaml
│   │   ├── children
│   │   │   ├── control-plane-dev.yaml
│   │   │   ├── control-plane-prod.yaml
│   │   │   ├── control-plane-staging.yaml
│   │   │   ├── gatekeeper.yaml
│   │   │   ├── monitoring.yaml
│   │   │   ├── umbrella-dev.yaml
│   │   │   ├── umbrella-prod.yaml
│   │   │   └── umbrella-staging.yaml
│   │   └── image-updater
│   │       └── README.md
│   ├── overlays
│   │   ├── dex
│   │   │   └── argocd-cm.yaml
│   │   └── sso
│   │       ├── argocd-cm.yaml
│   │       └── argocd-secret.yaml
│   └── projects
│       ├── dev.yaml
│       ├── prod.yaml
│       ├── rbac
│       │   └── argocd-rbac-cm.yaml
│       └── staging.yaml
├── assurance
│   ├── assurance.py
│   ├── cas.py
│   ├── errors.py
│   ├── example_run.py
│   ├── ops_registry.py
│   ├── respond_text.py
│   ├── signing.py
│   ├── tests
│   │   └── test_assurance_kernel.py
│   └── validators.py
├── attic
│   ├── api
│   │   └── http_api.py
│   ├── app
│   │   └── http_api.py
│   └── engine
│       └── http_api.py
├── audit
│   ├── audit_log.py
│   ├── cas.py
│   ├── ledger_sync.py
│   ├── ledger.py
│   ├── log.py
│   ├── merkle_log.py
│   ├── min_evidence.py
│   ├── provenance_store.py
│   └── signing.py
├── bridge
│   └── realtime_to_ui.py
├── broker
│   ├── __init__.py
│   ├── bus.py
│   ├── policy.py
│   ├── prior_broker.py
│   ├── stream_bus.py
│   ├── stream.py
│   ├── streams.py
│   └── ws_server.py
├── capabilities
│   ├── db_memory.py
│   ├── fs_read.py
│   ├── http_fetch.py
│   ├── manager.py
│   ├── more_mappings.py
│   └── synth.py
├── caps
│   ├── gpu_dispatch.py
│   ├── queue.py
│   ├── sqlite_sandbox.py
│   └── tasks
│       └── basic.py
├── cas
│   └── store.py
├── cli
│   └── grounded_cli.py
├── common
│   ├── errors.py
│   ├── exec.py
│   ├── provenance.py
│   └── ws_progress.py
├── compute
│   ├── backends.py
│   ├── ops.py
│   └── registry.py
├── config
│   ├── imu.json
│   ├── imu.local.json
│   ├── policy.json
│   ├── readme
│   └── runtime.json
├── contracts
│   ├── adapters.py
│   ├── base.py
│   ├── errors.py
│   └── schema.py
├── core
│   ├── __init__.py
│   ├── contracts
│   │   ├── __init__.py
│   │   └── verifier.py
│   └── vm
│       ├── __init__.py
│       └── vm.py
├── db
│   ├── contracts.py
│   ├── sandbox_multi.py
│   ├── sandbox_sqlite.py
│   ├── sandbox.py
│   └── strict_repo.py
├── demos
│   ├── android_build_end2end.py
│   ├── android_ios_build.py
│   ├── cuda_job_end2end.py
│   ├── docker_cosign_demo.py
│   ├── ios_build_end2end.py
│   ├── k8s_deploy_end2end.py
│   ├── pack_a_android_end2end.py
│   ├── pack_a_cuda_end2end.py
│   ├── pack_a_ios_end2end.py
│   ├── pack_a_k8s_end2end.py
│   ├── pack_a_unity_end2end.py
│   └── unity_k8s_e2e.py
├── device
│   ├── caps_device.py
│   └── policy.py
├── dist
│   ├── crdt.py
│   ├── dist_health.py
│   ├── job_queue.py
│   ├── lease_quorum.py
│   ├── raft_lite.py
│   ├── replication.py
│   ├── router.py
│   ├── service_registry.py
│   └── worker.py
├── distributed
│   ├── __init__.py
│   ├── crdt.py
│   └── raft.py
├── docker
│   ├── api
│   │   └── Dockerfile
│   ├── prod
│   │   ├── api
│   │   │   └── Dockerfile
│   │   ├── ui
│   │   │   └── Dockerfile
│   │   └── ws
│   │       └── Dockerfile
│   ├── ui
│   │   └── Dockerfile
│   └── ws
│       └── Dockerfile
├── docker-compose.yml
├── docs
│   ├── diagrams
│   │   └── architecture.md
│   ├── governance
│   │   └── slo.md
│   ├── index.md
│   └── runbooks
│       ├── deploy.md
│       └── emergency.md
├── engine
│   ├── __init__.py
│   ├── __main__.py
│   ├── ab_selector.py
│   ├── adapter_registry.py
│   ├── adapter_router.py
│   ├── adapter_runner.py
│   ├── adapter_types.py
│   ├── adapters
│   │   └── facade.py
│   ├── adapters.py
│   ├── agent_emit.py
│   ├── alerts.py
│   ├── artifacts
│   │   └── registry.py
│   ├── async_sandbox.py
│   ├── audit_log.py
│   ├── audit_rollup.py
│   ├── auto_remediation.py
│   ├── blueprints
│   │   ├── __init__.py
│   │   ├── ci_github_actions.py
│   │   ├── generic_backend.py
│   │   ├── iac_terraform.py
│   │   ├── market_scan.py
│   │   ├── observability.py
│   │   ├── registry.py
│   │   └── training_backend.py
│   ├── bootstrap.py
│   ├── build_orchestrator - pro.py
│   ├── build_orchestrator.py
│   ├── cache
│   │   └── merkle_cache.py
│   ├── canary_autotune.py
│   ├── canary_controller.py
│   ├── capabilities
│   │   ├── installers.py
│   │   └── registry.py
│   ├── capability_registry.py
│   ├── capability_wrap.py
│   ├── caps_db.py
│   ├── caps_distributed.py
│   ├── caps_realtime.py
│   ├── caps_ui.py
│   ├── cas_sign.py
│   ├── cas_store.py
│   ├── cas_verify.py
│   ├── cleanup
│   │   └── gc.py
│   ├── closed_loop.py
│   ├── config.py
│   ├── consistency_graph_weighted.py
│   ├── consistency_graph.py
│   ├── consistency_guard.py
│   ├── consistency.py
│   ├── contracts_gate.py
│   ├── contracts_policy.py
│   ├── convergence.py
│   ├── debug
│   │   └── fault_localizer.py
│   ├── domain_kb.py
│   ├── enforcement.py
│   ├── enofrcerer.py
│   ├── errors.py
│   ├── events.py
│   ├── evidence_freshness.py
│   ├── evidence_middleware.py
│   ├── exec_api.py
│   ├── explore_policy_ctx.py
│   ├── explore_policy.py
│   ├── explore_state.py
│   ├── fallbacks.py
│   ├── gates
│   │   ├── distributed_gate.py
│   │   ├── grounding_gate.py
│   │   ├── privacy_gate.py
│   │   ├── runtime_budget.py
│   │   ├── slo_gate.py
│   │   ├── streaming_gate.py
│   │   └── ui_gate.py
│   ├── grounding_gate.py
│   ├── guard_all.py
│   ├── guard_enforce.py
│   ├── guards.py
│   ├── hooks_policy.py
│   ├── hooks.py
│   ├── integrations_registry.py
│   ├── intent_to_spec.py
│   ├── json_diff.py
│   ├── key_delegation.py
│   ├── keychain_manager.py
│   ├── kpi_regression.py
│   ├── learn_store.py
│   ├── learn.py
│   ├── llm
│   │   ├── bandit_selector.py
│   │   ├── cache_integrations.py
│   │   └── cache.py
│   ├── llm_client.py
│   ├── llm_gateway-pro.py
│   ├── llm_gateway.py
│   ├── metrics
│   │   └── jsonl.py
│   ├── metrics_watcher.py
│   ├── micro_split.py
│   ├── orchestrator
│   │   ├── __init__.py
│   │   ├── universal_orchestrator.py
│   │   └── universal_planner.py
│   ├── pareto.py
│   ├── perf_sla.py
│   ├── phi_budget.py
│   ├── phi_multi_context.py
│   ├── phi_multi.py
│   ├── phi.py
│   ├── pipeline_bindings.py
│   ├── pipeline_default.py
│   ├── pipeline_events.py
│   ├── pipeline_multi.py
│   ├── pipeline_respond_hook.py
│   ├── pipeline.py
│   ├── pipelines
│   │   ├── compat.py
│   │   ├── orchestrator.py
│   │   └── shims.py
│   ├── plugin_api.py
│   ├── plugin_registry.py
│   ├── policy_compiler.py
│   ├── policy_ctx.py
│   ├── policy_drilldown.py
│   ├── policy_enforcer.py
│   ├── policy_guard.py
│   ├── policy_overrides.py
│   ├── policy.py
│   ├── prebuild
│   │   ├── adapter_builder.py
│   │   └── tool_acquisition.py
│   ├── progress.py
│   ├── prompt_builder.py
│   ├── provenance_gate.py
│   ├── provenance.py
│   ├── quarantine.py
│   ├── quorum_verify.py
│   ├── realtime_and_dist.py
│   ├── register_adapters.py
│   ├── registry.py
│   ├── reputation.py
│   ├── research
│   │   └── hypothesis_lab.py
│   ├── respond_bridge.py
│   ├── respond_guard.py
│   ├── respond_strict.py
│   ├── respond.py
│   ├── rollout_guard.py
│   ├── rollout_orchestrator.py
│   ├── rollout_quorum_gate.py
│   ├── runtime
│   │   ├── approvals.py
│   │   └── consent_store.py
│   ├── runtime_bridge.py
│   ├── runtime_guard.py
│   ├── self_heal
│   │   └── auto_pr.py
│   ├── snapshots
│   │   └── state_snapshot.py
│   ├── spec_refiner.py
│   ├── strict_grounded.py
│   ├── strict_mode.py
│   ├── synthesis_pipeline_ (ph.74).py
│   ├── synthesis_pipeline_(ph. 105~).py
│   ├── synthesis_pipeline.py
│   ├── testgen
│   │   ├── kpi_cases.py
│   │   ├── runtime_cases.py
│   │   └── synth_tests.py
│   ├── trust_tiers.py
│   ├── user_conflict_gate.py
│   ├── user_context_bridge.py
│   ├── user_scope.py
│   ├── verifier_km.py
│   ├── verifier.py
│   └── verify_bundle.py
├── evidence
│   ├── __init__.py
│   └── cas.py
├── examples
│   ├── adapters.usage.py
│   ├── android_build_example.py
│   ├── cuda_compile_example.py
│   ├── index.html
│   ├── ios_build.example.py
│   ├── k8s_apply_example.py
│   ├── progress_and_timeline.json
│   ├── rounded_fact_check.py
│   ├── run_adapters.py
│   ├── run_realtime_demo.py
│   ├── ui
│   │   └── index.html
│   ├── unity_build_example.py
│   ├── unity_k8s_flow.yaml
│   ├── unity_to_k8s_e2e.py
│   ├── unity_to_k8s_pipeline.py
│   ├── usage_android_ios_unity_cuda_k8s.py
│   └── usage_snippets.md 
├── exec
│   ├── cells.py
│   ├── detect.py
│   ├── errors.py
│   ├── languages
│   │   ├── cpp_runner.py
│   │   ├── csharp_runner.py
│   │   ├── go_runner.py
│   │   ├── java_runner.py
│   │   ├── node_runner.py
│   │   ├── python_runner.py
│   │   └── rust_runner.py
│   ├── select.py
│   └── simple_runner.py
├── executor
│   ├── __init__.py
│   ├── policy.py
│   ├── policy.yaml
│   └── sandbox.py
├── gitops
│   └── utils.py
├── governance
│   ├── __init__.py
│   ├── ab_verify.py
│   ├── canary_rollout.py
│   ├── enforcement.py
│   ├── policy.py
│   ├── proof_of_convergence.py
│   ├── slo_gate.py
│   └── user_policy.py
├── gpu
│   ├── __init__.py
│   └── runtime.py
├── grace
│   └── grace_manager.py
├── grounded
│   ├── __init__.py
│   ├── api_gate.py
│   ├── audit.py
│   ├── auto_patch.py
│   ├── claims.py
│   ├── consistency.py
│   ├── contradiction_policy.py
│   ├── evidence_contracts.py
│   ├── evidence_policy.py
│   ├── evidence_store.py
│   ├── fact_gate.py
│   ├── gate.py
│   ├── http_verifier.py
│   ├── ontradiction_resolution.py
│   ├── personal_evidence.py
│   ├── policy_overrides.py
│   ├── provenance_confidence.py
│   ├── provenance_sink.py
│   ├── provenance_store.py
│   ├── provenance.py
│   ├── runtime_sample.py
│   ├── schema_consistency.py
│   ├── source_policy.py
│   ├── trust.py
│   ├── ttl.py
│   ├── type_system.py
│   ├── validators.py
│   └── value_checks.py
├── grounding
│   ├── providers.py
│   ├── retrievers.py
│   └── validators_extra.py
├── guard
│   └── anti_regression.py
├── hardware
│   ├── __init__.py
│   └── gpio.py
├── helm
│   ├── control-plane
│   │   ├── Chart.yaml
│   │   ├── prometheusrule-alerts.yaml
│   │   ├── templates
│   │   │   ├── _helpers.tpl
│   │   │   ├── configmap-ui.yaml
│   │   │   ├── deployment-api.yaml
│   │   │   ├── deployment-ui.yaml
│   │   │   ├── deployment-ws.yaml
│   │   │   ├── gatekeeper-constraint.yaml
│   │   │   ├── gatekeeper-constrainttemplate.yaml
│   │   │   ├── grafana-dashboards-cm.yaml
│   │   │   ├── hooks
│   │   │   │   ├── postsync-helmtest-rollback.yaml
│   │   │   │   └── postsync-synthetics-rollback.yaml
│   │   │   ├── hpa.yaml
│   │   │   ├── ingress.yaml
│   │   │   ├── k6-configmap.yaml
│   │   │   ├── networkpolicy.yaml
│   │   │   ├── NOTES.txt
│   │   │   ├── pdb.yaml
│   │   │   ├── podsecurity-context.yaml
│   │   │   ├── prometheusrule-gatekeeper.yaml
│   │   │   ├── sa_rbac.yaml
│   │   │   ├── service.yaml
│   │   │   └── servicemonitor.yaml
│   │   ├── values.production.yaml
│   │   └── values.yaml
│   └── umbrella
│       ├── chart.yaml
│       ├── gating-alerts.yaml
│       ├── templates
│       │   ├── alertmanager-configmap-templates.yaml
│       │   ├── external-secrets-argocd-dex.yaml
│       │   ├── external-secrets-argocd-oidc.yaml
│       │   ├── external-secrets-store.yaml
│       │   ├── external-secrets.yaml
│       │   ├── gating-argocd-dex.yaml
│       │   ├── gating-oidc-grafana.yaml
│       │   ├── gating-secrets-exist.yaml
│       │   └── gating.yaml
│       ├── values.alerts.advanced.yaml
│       ├── values.alerts.pagers.yaml
│       ├── values.alerts.yaml
│       ├── values.dev.yaml
│       ├── values.oidc.yaml
│       ├── values.prod.yaml
│       ├── values.secrets.yaml
│       ├── values.staging.yaml
│       └── values.yaml
├── http
│   ├── api.py
│   └── sse_api.py
├── identity
│   └── profile_store.py
├── IMU.command
├── infra
│   ├── artifact_server.py
│   └── pulumi
│       ├── alerts-app-patch
│       │   ├── index.ts
│       │   ├── package.json
│       │   ├── Pulumi.yaml
│       │   └── tsconfig.json
│       └── oidc-secrets
│           ├── package.json
│           └── Pulumi.yaml
├── integration
│   ├── adapter_wrap.py
│   ├── bridge_exec.py
│   ├── compile_c.py
│   ├── example_bridge_run.py
│   └── llm_client.py
├── knowledge
│   ├── __init__.py
│   └── tools_store.py
├── kpi
│   ├── aggregate.py
│   ├── policy_adapter.py
│   └── score.py
├── learning
│   ├── __init__.py
│   ├── event_log.py
│   ├── event_sink.py
│   ├── example_learn.py
│   ├── pattern_store.py
│   ├── policy_learner.py
│   └── supervisor.py
├── Makefile
├── metrics
│   └── aggregate.py
├── middleware
│   └── evidence_scope.py
├── mkdocs.yml
├── monitoring
│   └── grafana
│       ├── dashboards
│       │   ├── imu_allowed_diffs.json
│       │   ├── imu_api.json
│       │   ├── imu_gate_trends.json
│       │   ├── imu_gatekeeper.json
│       │   ├── imu_kind_smoke_slo.json
│       │   ├── imu_policy_drilldown.json
│       │   ├── imu_scheduler.json
│       │   ├── imu_slo.json
│       │   └── imu_ws.json
│       └── README.md
├── next_phase.sh
├── obs
│   ├── __init__.py
│   ├── alerts.py
│   ├── kpi.py
│   └── tracing.py
├── observability
│   └── server.py
├── optimizer
│   └── phi.py
├── orchestration
│   ├── compose_workflow.py
│   ├── docker_compose.py
│   └── services.py
├── orchestrator
│   ├── consensus.py
│   ├── orchestrator.py
│   ├── registry.py
│   └── worker_runtime.py
├── package-lock.json
├── package.json
├── packaging
│   ├── html_bundle.py
│   └── packager.py
├── perf
│   ├── measure_ab.py
│   ├── measure.py
│   ├── monitor.py
│   └── p95.py
├── persistence
│   └── policy_store.py
├── pipeline
│   ├── __init__.py
│   └── synthesis.py
├── plugins
│   ├── compute
│   │   └── vector_ops.py
│   ├── db
│   │   └── sqlite_sandbox.py
│   └── ui
│       └── static_site.py
├── policy
│   ├── __init__.py
│   ├── adaptive.py
│   ├── auto_approval.py
│   ├── cite_or_silence.py
│   ├── enforce.py
│   ├── enforcement.py
│   ├── freshness_profile.py
│   ├── lint.py
│   ├── model.py
│   ├── policies.py
│   ├── policy_enforcer.py
│   ├── policy_engine.py
│   ├── policy_hotload.py
│   ├── policy_rules.py
│   ├── rbac.py
│   ├── rego
│   │   ├── external_dns.rego
│   │   ├── ingress_class.rego
│   │   ├── ingress_external_dns.rego
│   │   └── ingress_tls.rego
│   ├── ttl.py
│   ├── user_policy.py
│   └── user_subspace.py
├── privacy
│   ├── keystore.py
│   └── storage.py
├── program
│   └── orchestrator.py
├── program_sbx
│   ├── sbx_1igfy0wt
│   │   ├── in
│   │   │   └── app.py
│   │   ├── out
│   │   └── tmp
│   ├── sbx_5sxoxq9i
│   │   ├── in
│   │   │   └── app.py
│   │   ├── out
│   │   └── tmp
│   ├── sbx_6ra1ia4r
│   │   ├── in
│   │   │   └── app.py
│   │   ├── out
│   │   └── tmp
│   ├── sbx_i8fzb1m5
│   │   ├── in
│   │   │   ├── app.py
│   │   │   └── test_app.py
│   │   ├── out
│   │   └── tmp
│   ├── sbx_ktcn8_2i
│   │   ├── in
│   │   │   ├── app.py
│   │   │   └── test_app.py
│   │   ├── out
│   │   └── tmp
│   ├── sbx_nikqcp_z
│   │   ├── in
│   │   │   └── app.py
│   │   ├── out
│   │   └── tmp
│   ├── sbx_npyu45ol
│   │   ├── in
│   │   │   └── app.py
│   │   ├── out
│   │   └── tmp
│   ├── sbx_s0uz8qy2
│   │   ├── in
│   │   │   └── app.py
│   │   ├── out
│   │   └── tmp
│   ├── sbx_vkugr7rc
│   │   ├── in
│   │   │   ├── app.py
│   │   │   └── test_app.py
│   │   ├── out
│   │   └── tmp
│   ├── sbx_vp41ckv9
│   │   ├── in
│   │   │   └── app.py
│   │   ├── out
│   │   └── tmp
│   ├── sbx_x2z3ateb
│   │   ├── in
│   │   │   ├── app.py
│   │   │   └── test_app.py
│   │   ├── out
│   │   └── tmp
│   └── sbx_zoix6z8y
│       ├── in
│       │   └── app.py
│       ├── out
│       └── tmp
├── provenance
│   ├── artifact_index.py
│   ├── audit.py
│   ├── ca_store.py
│   ├── cas.py
│   ├── castore.py
│   ├── envelope.py
│   ├── keyring.py
│   ├── policy.py
│   ├── provenance.py
│   ├── runtime_lineage.py
│   ├── sign.py
│   ├── signer.py
│   ├── signing.py
│   ├── store.py
│   └── trust_registry.py
├── pytest.ini
├── realtime
│   ├── __init__.py
│   ├── backpressure.py
│   ├── integrations.py
│   ├── metrics_stream.py
│   ├── pmdeflate.py
│   ├── priority_bus.py
│   ├── protocol.py
│   ├── qos_broker.py
│   ├── run_tcp_server.py
│   ├── server.py
│   ├── streaming.py
│   ├── strict_sink.py
│   ├── strict_ws.py
│   ├── tcp_framed.py
│   ├── webrtc.py
│   ├── ws_broker.py
│   ├── ws_core.py
│   ├── ws_guarded_server.py
│   ├── ws_loopback.py
│   ├── ws_minimal.py
│   ├── ws_proto.py
│   ├── ws_push.py
│   └── ws_server.py
├── requirements.txt
├── rt
│   ├── async_runtime.py
│   └── queue.py
├── run_example.py
├── run_pipeline_multi.py
├── run_pipeline.py
├── run_sandboxes
├── run_server.py
├── run_servers.py
├── runtime
│   ├── async_sandbox.py
│   ├── metrics.py
│   ├── p95.py
│   └── sandbox.py
├── safe_progress
│   └── auto_rollout.py
├── sandbox
│   ├── fs_net.py
│   ├── fs.py
│   ├── limits.py
│   ├── net_class_rl.py
│   ├── net_client.py
│   ├── net_rl.py
│   └── session_rl.py
├── sbom
│   └── cyclonedx_demo.json
├── scripts
│   ├── allowed_diffs.yaml
│   ├── ci_metrics.sh
│   ├── diff_umbrella.py
│   ├── gen_alerts_values.sh
│   ├── gen_mermaid_from_values.py
│   ├── imu_up.sh
│   ├── kind_setup.sh
│   ├── one_button_platform.sh
│   ├── phase_audit.sh
│   ├── quick_deploy.sh
│   ├── smoke_all.sh
│   ├── smoke_kind_ci.sh
│   ├── smoke_kind.sh
│   ├── support_bundle.sh
│   ├── umbrella_smoke_kind_ci.sh
│   └── ws_synthetic_ci.py
├── secrets
│   └── sealed
│       └── alerts-slack-sealed.yaml
├── security
│   ├── ed25519_optional.py
│   ├── filesystem_policies.py
│   ├── fingerprint_report.py
│   ├── network_policies.py
│   ├── policy_rules.yaml
│   ├── policy.py
│   ├── response_signer.py
│   ├── sandbox.py
│   └── signing.py
├── self_improve
│   ├── ab_runner.py
│   ├── apply.py
│   ├── executors
│   │   ├── base.py
│   │   ├── db_executor.py
│   │   ├── guard_executor.py
│   │   └── ws_executor.py
│   ├── fix_plan.py
│   ├── patcher.py
│   ├── planner.py
│   ├── regression_guard.py
│   └── testgen.py
├── server
│   ├── __init__.py
│   ├── archive_api.py
│   ├── audit_ops.py
│   ├── boot_strict.py
│   ├── boot.py
│   ├── bootstrap.py
│   ├── bundles_api.py
│   ├── canary_auto_api.py
│   ├── canary_auto_policy.py
│   ├── canary_controller.py
│   ├── capabilities
│   │   ├── impl
│   │   │   ├── android.py
│   │   │   ├── cuda.py
│   │   │   ├── ffmpeg.py
│   │   │   ├── ios.py
│   │   │   ├── k8s.py
│   │   │   ├── unity.py
│   │   │   └── webrtc.py
│   │   ├── registry.py
│   │   └── types.py
│   ├── controlplane_deploy_api.py
│   ├── decision_log.py
│   ├── dialog
│   │   ├── intent.py
│   │   ├── memory_bridge.py
│   │   ├── planner.py
│   │   └── state.py
│   ├── emergency_api.py
│   ├── events
│   │   └── bus.py
│   ├── gatekeeper_api.py
│   ├── gatekeeper_client.py
│   ├── gh_status_api.py
│   ├── gitops_api.py
│   ├── gitops_checks_api.py
│   ├── gitops_guard_api.py
│   ├── helm_template_synth_api.py
│   ├── http_api.py
│   ├── job_runs.py
│   ├── k8s_ready.py
│   ├── k8s_template_synth_api.py
│   ├── key_admin_api.py
│   ├── main.py
│   ├── merge_guard_api.py
│   ├── metrics_api.py
│   ├── metrics_jobs_api.py
│   ├── pac_pipeline.py
│   ├── pipeline
│   │   └── run_adapter.py
│   ├── planning.py
│   ├── policy
│   │   └── enforcement.py
│   ├── policy_edit_api.py
│   ├── private_repo_fetch.py
│   ├── prom_anomaly.py
│   ├── prometheus_client.py
│   ├── provenance_api.py
│   ├── replay_api.py
│   ├── routers
│   │   ├── adapters_secure.py
│   │   ├── build_api.py
│   │   ├── cache_api.py
│   │   ├── chat_api -pro.py
│   │   ├── chat_api.py
│   │   ├── consent_api.py
│   │   ├── llm_api.py
│   │   ├── orchestrate_api.py
│   │   ├── prebuild_api.py
│   │   ├── program_api.py
│   │   ├── respond_api.py
│   │   └── user_memory.py
│   ├── runbook_api.py
│   ├── runbook_history.py
│   ├── runtime_init.py
│   ├── scheduler_api.py
│   ├── security
│   │   ├── audit.py
│   │   └── provenance.py
│   ├── state
│   │   └── ttl.py
│   ├── stream_gateway.py
│   ├── stream_policy_router.py
│   ├── stream_wfq_stats.py
│   ├── stream_wfq_ws.py
│   ├── stream_wfq.py
│   ├── supplychain_api.py
│   ├── supplychain_index_api.py
│   ├── synth_adapter_api.py
│   ├── synth_presets_api.py
│   ├── unified_archive_api.py
│   ├── user_memory.py
│   ├── webhooks_api.py
│   ├── ws_progress.py
│   └── ws.py
├── service_mesh
│   ├── health.py
│   ├── policy.py
│   └── router.py
├── services
│   ├── api
│   │   ├── app.py
│   │   ├── compute.py
│   │   ├── db.py
│   │   ├── models.py
│   │   ├── requirements.txt
│   │   ├── spec_loader.py
│   │   └── tests
│   │       └── test_app.py
│   ├── artifact_server.py
│   └── broker_ws.py
├── sitecustomize.py
├── sla
│   └── policy.py
├── storage
│   ├── cas.py
│   ├── provenance_store.py
│   ├── provenance.py
│   └── ttl_store.py
├── stream
│   └── broker.py
├── streaming
│   ├── broker.py
│   └── ws_server.py
├── streams
│   ├── backpressure.py
│   ├── broker_client.py
│   └── broker.py
├── synth
│   ├── canary_multi.py
│   ├── canary.py
│   ├── evidence_schemas.py
│   ├── generate_ab_explore.py
│   ├── generate_ab_prior.py
│   ├── generate_ab.py
│   ├── generate.py
│   ├── package.py
│   ├── plan.py
│   ├── rollout.py
│   ├── schema_validate.py
│   ├── specs_adapter.py
│   ├── specs.py
│   ├── test.py
│   ├── validators.py
│   └── verify.py
├── tests
│   ├── __init__.py
│   ├── _test_stage87_sandbox_limits.py
│   ├── benchmarks.py
│   ├── compose_stack.py
│   ├── conftest.py
│   ├── est_docs_mkdocs_yaml.py
│   ├── exec_cells.py
│   ├── external_validation.py
│   ├── grounding_strict.py
│   ├── integration_micro.py
│   ├── integration_workflow.py
│   ├── load_phi_rollout.py
│   ├── redcases
│   │   ├── values.bad-externaldns-disallowed-zone.yaml
│   │   ├── values.bad-externaldns-off.yaml
│   │   ├── values.bad-externaldns-provider.yaml
│   │   ├── values.bad-externaldns-zone.yaml
│   │   ├── values.bad-ingress-host-outside-zone.yaml
│   │   ├── values.bad-ingress-missing-externaldns-annotation.yaml
│   │   └── values.bad-ingress-no-tls.yaml
│   ├── run_unity_pipeline.py
│   ├── sandbox_io_net.py
│   ├── smoke.py
│   ├── test_adapters_and_policy.py
│   ├── test_adapters_b.py
│   ├── test_adapters_dry_run.py
│   ├── test_adapters_dryrun.py
│   ├── test_adapters_env.py
│   ├── test_adapters_pack_a.py
│   ├── test_adapters_packA.py
│   ├── test_adapters_requirements.py
│   ├── test_adapters_secure.py
│   ├── test_adapters_smoke.py
│   ├── test_adapters.py
│   ├── test_alert_pagers_values_and_gating.py
│   ├── test_alert_templates_deeplinks.py
│   ├── test_alerts_advanced_and_prbot.py
│   ├── test_alerts_templates_and_bot.py
│   ├── test_allowed_diffs_config.py
│   ├── test_allowed_diffs_dashboard_json.py
│   ├── test_allowed_diffs_env_yaml.py
│   ├── test_android_optional.py
│   ├── test_argocd_apps_yaml.py
│   ├── test_attest_verify_routes.py
│   ├── test_audit_cas_diff.py
│   ├── test_auto_approval.py
│   ├── test_auto_canary_gatekeeper.py
│   ├── test_auto_canary_merge_guard.py
│   ├── test_backpressure_adv.py
│   ├── test_backpressure_and_ui.py
│   ├── test_build_commands.py
│   ├── test_bundles_rbac.py
│   ├── test_canary_and_merge_guard.py
│   ├── test_canary_stages_and_freshness.py
│   ├── test_cas_and_diff_units.py
│   ├── test_cas_signing_and_profiles.py
│   ├── test_ci_and_hooks_files_exist.py
│   ├── test_ci_metrics_helper.py
│   ├── test_config_loader.py
│   ├── test_contracts_and_adapters.py
│   ├── test_contracts_strict.py
│   ├── test_controlplane_deploy_api.py
│   ├── test_dashboards_all_listed.py
│   ├── test_demos.py
│   ├── test_dex_overlay_and_mermaid_gen.py
│   ├── test_docker_optional.py
│   ├── test_dual_sign_and_rollup.py
│   ├── test_e2e_unity_k8s.py
│   ├── test_end2end_examples.py
│   ├── test_executer_policy_and_compile.py
│   ├── test_external_secrets_files.py
│   ├── test_gate_trends_rule_and_dashboard.py
│   ├── test_gatekeeper_dashboard_json.py
│   ├── test_gatekeeper_private_fetch.py
│   ├── test_gating_ingress_externaldns_flags.py
│   ├── test_generated_db-pg-backup.py
│   ├── test_generated_db-pg-restore.py
│   ├── test_generated_gpu-cuda-smi-log.py
│   ├── test_generated_gpu-nvml-metrics.py
│   ├── test_generated_helm_templates.py
│   ├── test_generated_helm-rollback.py
│   ├── test_generated_infra-ansible-apply.py
│   ├── test_generated_infra-ansible-galaxy.py
│   ├── test_generated_infra-terraform-apply.py
│   ├── test_generated_kpi_cases.py
│   ├── test_generated_queue-kafka-produce.py
│   ├── test_generated_queue-nats-publish.py
│   ├── test_generated_queue-rabbitmq-publish.py
│   ├── test_generated_queue-redis-subscribe.py
│   ├── test_generated_runtime_cases.py
│   ├── test_generated_scm-git-clone.py
│   ├── test_generated_storage-azure-blob-sync.py
│   ├── test_generated_storage-gcs-sync.py
│   ├── test_generated_storage-s3-sync.py
│   ├── test_gitops_api.py
│   ├── test_gitops_checks_and_policy_editor.py
│   ├── test_gitops_guard.py
│   ├── test_global)consistency_graph.py
│   ├── test_grafana_dashboards_json.py
│   ├── test_grounded_end2end.py
│   ├── test_grounded_enforced.py
│   ├── test_grounded_text_and_program.py
│   ├── test_guard_and_kpi_auto_remediation.py
│   ├── test_guard_runtime_kpi.py
│   ├── test_helm_templates_advanced.py
│   ├── test_helm_templates_profiles_and_cert.py
│   ├── test_hotload_and_runbook.py
│   ├── test_http_api_smoke.py
│   ├── test_http_sse_broker.py
│   ├── test_http_ui_live.py
│   ├── test_identity_privacy.py
│   ├── test_ios_optional.py
│   ├── test_jobs_metrics_and_router.py
│   ├── test_k6_ws_present.py
│   ├── test_k6_ws_publish_flow.py
│   ├── test_k8s_manifest_provenance.py
│   ├── test_k8s_optional.py
│   ├── test_k8s_template_synth.py
│   ├── test_key_admin_and_archive.py
│   ├── test_kind_ci_workflow_and_scripts.py
│   ├── test_kind_smoke_scripts.py
│   ├── test_kpi_gate.py
│   ├── test_kpi_regression.py
│   ├── test_loki_dashboard_file.py
│   ├── test_mermaid_diagram_md_exists.py
│   ├── test_multi_user_policy.py
│   ├── test_oidc_files_and_emergency.py
│   ├── test_one_button_and_slo_dashboard.py
│   ├── test_p95_wfq_ws.py
│   ├── test_pack_a_dry_end2end.py
│   ├── test_pack_a_end2end.py
│   ├── test_pdb_and_spread_present.py
│   ├── test_perf_and_grounded.py
│   ├── test_perf_sla_and_autotune.py
│   ├── test_pipeline_respond_hook.py
│   ├── test_pipeline_with_policy.py
│   ├── test_policies_and_adapters.py
│   ├── test_policy_and_provenance.py
│   ├── test_policy_provenance_and_adapters.py
│   ├── test_policy_ttl_provenance.py
│   ├── test_pr_notify_workflow_exists.py
│   ├── test_prom_ready_status.py
│   ├── test_provenance_and_policy.py
│   ├── test_provenance_ed25519.py
│   ├── test_provenance_store.py
│   ├── test_provenance.py
│   ├── test_pulumi_and_generators.py
│   ├── test_qos_rate_limit.py
│   ├── test_quick_deploy_script.py
│   ├── test_realtime_throttle.py
│   ├── test_realtime_ws_push.py
│   ├── test_redcases_files_exist.py
│   ├── test_replay_archive.py
│   ├── test_reputation_quarantine_slo.py
│   ├── test_request_and_continue.py
│   ├── test_respond_integration.py
│   ├── test_runtime_drift_and_baseline.py
│   ├── test_runtime_guard_remedies.py
│   ├── test_runtime_lineage.py
│   ├── test_runtime_p95_wfq.py
│   ├── test_sandbox_db_ui.py
│   ├── test_scheduler_api_basic.py
│   ├── test_scheduler_tasks.py
│   ├── test_scoped_delegations_and_quorum_gate.py
│   ├── test_smoke_all_script.py
│   ├── test_spike_status_decisions.py
│   ├── test_sqlite_sandbox.py
│   ├── test_stage_90_raft_loopback.py
│   ├── test_stage101_strict_and_consistency.py
│   ├── test_stage102_strict_everywhere_and_user_overrides.py
│   ├── test_stage103_end2end_ui_realtime_db.py
│   ├── test_stage104_ui_advanced.py
│   ├── test_stage105_realtime_strict.py
│   ├── test_stage106_ui_stream.py
│   ├── test_stage38_interactive.py
│   ├── test_stage39_accessibility.py
│   ├── test_stage40_trust_policy.py
│   ├── test_stage41_contract_trust_and_fetch.py
│   ├── test_stage42_consistency_fail_local.py
│   ├── test_stage42_consistency_ok.py
│   ├── test_stage43_resolution_fail.py
│   ├── test_stage43_resolution_ok.py
│   ├── test_stage44_gated_rollout_ok.py
│   ├── test_stage44_regression_block.py
│   ├── test_stage45_personalized_policies.py
│   ├── test_stage46_engine_integration.py
│   ├── test_stage46_user_consolidation.py
│   ├── test_stage47_consensus_and_routing.py
│   ├── test_stage47_realtime_queue.py
│   ├── test_stage48_multi_component_rollout.py
│   ├── test_stage49_plugins.py
│   ├── test_stage50_audit_and_provenance.py
│   ├── test_stage50_crdt_replication.py
│   ├── test_stage50_official_api_gate.py
│   ├── test_stage50_pipeline_with_api_gate.py
│   ├── test_stage51_ledger_sync_hmac.py
│   ├── test_stage51_min_evidence_gate.py
│   ├── test_stage52_pipeline_user_policy.py
│   ├── test_stage52_user_consciousness.py
│   ├── test_stage53_conflict_gate_in_pipeline.py
│   ├── test_stage53_user_sync.py
│   ├── test_stage54_db_queue_caps.py
│   ├── test_stage54_realtime_runtime.py
│   ├── test_stage54_runtime_budget_gate.py
│   ├── test_stage55_orchestration.py
│   ├── test_stage56_service_mesh.py
│   ├── test_stage57_ui_and_gpu.py
│   ├── test_stage58_grounding.py
│   ├── test_stage59_user_consciousness.py
│   ├── test_stage60_distributed.py
│   ├── test_stage61_realtime.py
│   ├── test_stage61b_realtime_deflate.py
│   ├── test_stage62_compute_ui_packaging.py
│   ├── test_stage63_provenance_and_bundle.py
│   ├── test_stage64_grounding_gate.py
│   ├── test_stage65_ws_guarded_e2e.py
│   ├── test_stage66_sla_and_observability.py
│   ├── test_stage67_self_sustaining.py
│   ├── test_stage68_auto_fix.py
│   ├── test_stage69_continuous_auto_fix.py
│   ├── test_stage70_hooks_and_grounding.py
│   ├── test_stage71_ws_adaptive_and_strict.py
│   ├── test_stage72_user_grounded_defaults.py
│   ├── test_stage73_end2end_claims_everywhere.py
│   ├── test_stage74_capabilities_guarded.py
│   ├── test_stage75_synthesis_guarded.py
│   ├── test_stage76_ab_bestofall.py
│   ├── test_stage77_convergence.py
│   ├── test_stage78_prior_guided.py
│   ├── test_stage79_prior_exploration.py
│   ├── test_stage80_phi_multi.py
│   ├── test_stage81_context_pareto.py
│   ├── test_stage82_explore_adaptive.py
│   ├── test_stage83_provenance_confidence.py
│   ├── test_stage84_guard_enforce.py
│   ├── test_stage85_official_verifiers.py
│   ├── test_stage86_multi_tenant_privacy.py
│   ├── test_stage87_sandbox_limits.py
│   ├── test_stage88_async_budget.py
│   ├── test_stage89_ws_loopback.py
│   ├── test_stage91_db_sandbox.py
│   ├── test_stage92_ui_sandbox.py
│   ├── test_stage92c_ui_extended.py
│   ├── test_stage93_device_caps.py
│   ├── test_stage93a_ui_grid_filter_sort.py
│   ├── test_stage94b_ui_advanced_grid_table.py
│   ├── test_stage95_signing.py
│   ├── test_stage96_provenance_gate_and_cli.py
│   ├── test_stream_advanced.py
│   ├── test_supplychain_api.py
│   ├── test_supplychain_index.py
│   ├── test_synth_adapter_flow.py
│   ├── test_synthesis_end2end.py
│   ├── test_synthetics_and_diff_files_exist.py
│   ├── test_trust_and_consistency.py
│   ├── test_trust_delegation_and_quorum.py
│   ├── test_umbrella_chart_and_argocd.py
│   ├── test_umbrella_gating_and_annotations.py
│   ├── test_umbrella_smoke_ci_files.py
│   ├── test_unity_k8s_e2e.py
│   ├── test_unity_optional.py
│   ├── test_values_prod_yaml.py
│   ├── test_verify_and_nearmiss.py
│   ├── test_webhooks_pac.py
│   ├── test_weighted_consistency.py
│   ├── test_ws_backpressure_and_topics.py
│   ├── test_ws_broker_local.py
│   ├── test_ws_synthetic_script_and_alerts.py
│   ├── test100_consistency_and_negative.py
│   ├── test101_schema_consistency.py
│   ├── test102_runtime_consistency.py
│   ├── test97_policy_and_signing.py
│   ├── test98_adaptive_and_chain.py
│   ├── test99_freshness_and_audit.py
│   ├── tests
│   │   ├── test_proof_chain.py
│   │   └── test_respond_integration.py
│   ├── user_profile.py
│   ├── vm_concurrency.py
│   └── vm_subroutines.py
├── tools
│   ├── auto_install.py
│   ├── imu_audit_dump.py
│   ├── imu_keygen.py
│   ├── imu_policy_tune.py
│   ├── imu_rotate_key.py
│   ├── imu_trust.py
│   ├── imu_verify.py
│   ├── orchestrate.py
│   └── test_orchestrator.py
├── TREE.md
├── ui
│   ├── __init__.py
│   ├── accessibility_gate.py
│   ├── accessibility.py
│   ├── auto_canary.html
│   ├── bundles.html
│   ├── canary.html
│   ├── dashboard.html
│   ├── deploy_control_plane.html
│   ├── desktop.py
│   ├── dsl_runtime_rt.py
│   ├── dsl.py
│   ├── emergency.html
│   ├── example.html
│   ├── forms.py
│   ├── game.py
│   ├── gatekeeper.html
│   ├── gen_frontend.py
│   ├── gitops_guard.html
│   ├── gitops.html
│   ├── helm_templates_visual.html
│   ├── helm_templates.html
│   ├── index.html
│   ├── introspect.py
│   ├── jobs_dashboard.html
│   ├── k8s_templates.html
│   ├── keys.html
│   ├── metrics.html
│   ├── mobile.py
│   ├── package.py
│   ├── policy_live.html
│   ├── proofs_view.js
│   ├── public
│   │   ├── index.html
│   │   └── progress.js
│   ├── render.py
│   ├── replay.html
│   ├── rt_client.py
│   ├── runbook.html
│   ├── sbom.html
│   ├── scheduler.html
│   ├── schema_compose.py
│   ├── schema_extract.py
│   ├── static
│   │   ├── app.html
│   │   └── progress.html
│   ├── static_pack.py
│   ├── synth_examples.html
│   ├── synth_wizard.html
│   ├── synth.html
│   ├── timeline_filters.html
│   ├── toolkits_bridge.py
│   ├── ui_dsl_runtime.js
│   ├── unified_static.py
│   ├── web
│   │   ├── client_widget.js
│   │   └── demo.html
│   ├── web.py
│   └── webhooks_console.html
├── ui_dsl
│   ├── advanced_components.py
│   ├── client_ws.js
│   ├── compiler.py
│   ├── components
│   │   ├── streams.ts
│   │   ├── timeline.js
│   │   └── widgets.tsx
│   ├── index.html
│   ├── live_bindings.py
│   ├── provenance.py
│   ├── renderer_v2.py
│   ├── runtime
│   │   ├── client.js
│   │   ├── live_bind.tsx
│   │   └── stream_timeline.js
│   ├── runtime_live.py
│   ├── runtime_stream_bindings.js
│   ├── runtime.js
│   ├── schema.json
│   ├── static_signer.py
│   ├── stream_components.py
│   ├── stream_widgets.js
│   ├── stream_widgets.py
│   ├── strict_renderer.py
│   ├── style.css
│   ├── versioning.py
│   └── widgets
│       └── streams.py
├── ui_runtime
│   └── stream_widgets.js
├── user
│   ├── __init__.py
│   ├── auth.py
│   ├── consciousness.py
│   ├── consolidation.py
│   ├── crypto_store.py
│   ├── memory_state.py
│   ├── semvec.py
│   └── util.py
├── user_model
│   ├── conflict_resolution.py
│   ├── consciousness.py
│   ├── consent.py
│   ├── consolidation.py
│   ├── crypto_store.py
│   ├── crypto_utils.py
│   ├── emotion.py
│   ├── event_crdt.py
│   ├── identity.py
│   ├── intent.py
│   ├── memory_store.py
│   ├── memory.py
│   ├── model.py
│   ├── policies.py
│   ├── policy.py
│   ├── profile_store.py
│   ├── routing.py
│   ├── semantic_store.py
│   ├── subject.py
│   ├── sync_protocol.py
│   └── user_policy_bridge.py
├── var
│   ├── audit
│   └── prov
├── verifiers
│   ├── official_gate.py
│   ├── official_registry.py
│   └── official_verify.py
├── watcher
│   └── auto_fix_daemon.py
└── webui
    ├── app.js
    └── index.html

273 directories, 1279 files

