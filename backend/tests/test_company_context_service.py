import json
from types import SimpleNamespace

import pytest

from app.models.workspace import CompanyProfile
from app.services.llm.types import LLMOrchestrationError, LLMResponse, LLMStage, ModelAttemptTrace
from app.services.company_context import (
    _capability_phrase_from_text,
    _merge_reasoned_expansion_brief,
    _report_source_label,
    _sourcing_brief_reasoning_prompt,
    apply_scope_review_decisions,
    build_expansion_artifacts,
    build_expansion_report_artifact,
    build_company_context_artifacts,
    build_sourcing_report_artifact,
    build_context_pack_v2,
    derive_discovery_scope_hints,
    derive_scope_review_payload,
    normalize_expansion_brief,
)


@pytest.fixture(autouse=True)
def _disable_live_llm(monkeypatch):
    monkeypatch.setattr(
        "app.services.company_context.get_settings",
        lambda: SimpleNamespace(
            gemini_api_key="",
            openai_api_key="",
            anthropic_api_key="",
        ),
    )


def _build_profile() -> CompanyProfile:
    return CompanyProfile(
        workspace_id=1,
        buyer_company_url="https://acme.example.com",
        comparator_seed_urls=["https://comp-one.example.com"],
        supporting_evidence_urls=["https://acme.example.com/customers"],
        comparator_seed_summaries={"https://comp-one.example.com": "Comparable analytics vendor."},
        geo_scope={"region": "US", "include_countries": ["US"], "exclude_countries": ["RU"]},
        context_pack_json={
            "sites": [
                {
                    "url": "https://acme.example.com",
                    "company_name": "Acme",
                    "summary": "Portfolio analytics and reporting for fund operations teams.",
                    "signals": [
                        {
                            "type": "capability",
                            "value": "Portfolio analytics",
                            "source_url": "https://acme.example.com/platform",
                        },
                        {
                            "type": "service",
                            "value": "Implementation services",
                            "source_url": "https://acme.example.com/services",
                        },
                        {
                            "type": "integration",
                            "value": "API integrations",
                            "source_url": "https://acme.example.com/integrations",
                        },
                    ],
                    "customer_evidence": [
                        {
                            "name": "Northwind Capital",
                            "source_url": "https://acme.example.com/customers",
                        }
                    ],
                    "pages": [
                        {
                            "url": "https://acme.example.com/platform",
                            "title": "Platform",
                            "signals": [],
                            "customer_evidence": [],
                        }
                    ],
                }
            ]
        },
        product_pages_found=5,
    )


def _build_expansion_payload(profile: CompanyProfile, payload: dict) -> dict:
    return build_expansion_artifacts(
        profile,
        sourcing_brief=payload["sourcing_brief"],
        taxonomy_nodes=payload["taxonomy_nodes"],
    )


def test_build_company_context_artifacts_generates_source_documents_and_brief():
    profile = _build_profile()
    payload = build_company_context_artifacts(profile)
    expansion_payload = _build_expansion_payload(profile, payload)

    assert payload["source_pills"]
    assert payload["buyer_evidence"]["status"] == "sufficient"
    assert payload["buyer_evidence"]["used_for_inference"] is True
    assert payload["context_pack_v2"]["version"] == "v2"
    assert "taxonomy_nodes" in payload
    assert "sourcing_brief" in payload
    assert "expansion_inputs" in payload
    assert payload["sourcing_brief"]["source_summary"]
    assert payload["sourcing_brief"]["named_customer_proof"]
    assert payload["lens_seeds"]

    sourcing_report = build_sourcing_report_artifact(
        sourcing_brief=payload["sourcing_brief"],
        source_documents=[],
        context_pack_v2=payload["context_pack_v2"],
        confirmed_at=payload["confirmed_at"],
    )
    expansion_report = build_expansion_report_artifact(
        source_company=(payload["sourcing_brief"] or {}).get("source_company") or {},
        expansion_brief=expansion_payload["expansion_brief"],
        source_documents=[],
        context_pack_v2=payload["context_pack_v2"],
        confirmed_at=(expansion_payload["expansion_brief"] or {}).get("confirmed_at"),
    )

    assert sourcing_report["artifact_type"] == "report_artifact"
    assert sourcing_report["report_kind"] == "sourcing_brief"
    assert sourcing_report["sections"]
    assert expansion_report["report_kind"] == "expansion_brief"


def test_sourcing_report_summary_uses_named_proof_sources_when_taxonomy_sources_are_missing():
    payload = build_company_context_artifacts(_build_profile())
    brief = json.loads(json.dumps(payload["sourcing_brief"]))
    for key in ("customer_nodes", "workflow_nodes", "capability_nodes", "delivery_or_integration_nodes"):
        brief[key] = []

    sourcing_report = build_sourcing_report_artifact(
        sourcing_brief=brief,
        source_documents=[],
        context_pack_v2=payload["context_pack_v2"],
        confirmed_at=payload["confirmed_at"],
    )

    summary_section = sourcing_report["sections"][0]
    summary_sentence = summary_section["blocks"][0]["sentences"][0]
    assert summary_sentence["citation_pill_ids"]


def test_sourcing_report_summary_prefers_first_party_sources_over_secondary_pages():
    sourcing_report = build_sourcing_report_artifact(
        sourcing_brief={
            "source_company": {"name": "Hublo", "website": "https://hublo.com"},
            "source_summary": "Hublo runs workforce management for hospitals.",
            "customer_nodes": [],
            "workflow_nodes": [],
            "capability_nodes": [],
            "delivery_or_integration_nodes": [],
            "named_customer_proof": [],
            "partner_integration_proof": [],
            "secondary_evidence_proof": [
                {
                    "url": "https://play.google.com/store/apps/details?id=fr.medgo.medgo1",
                    "publisher": "play.google.com",
                    "claim_type": "workflow_description",
                }
            ],
        },
        source_documents=[
            {
                "id": "hublo_primary",
                "url": "https://hublo.com/",
                "name": "Hublo",
                "publisher_channel": "primary",
                "publisher_type": "source_company",
                "evidence_tier": "primary",
            }
        ],
        context_pack_v2={},
        confirmed_at=None,
    )

    summary_sentence = sourcing_report["sections"][0]["blocks"][0]["sentences"][0]
    source_ids = summary_sentence["citation_pill_ids"]
    sources_by_id = {item["id"]: item for item in sourcing_report["sources"]}

    assert source_ids
    assert sources_by_id[source_ids[0]]["url"] == "https://hublo.com/"


def test_expansion_report_summary_dedupes_repeated_citations():
    expansion_brief = {
        "adjacent_capabilities": [
            {
                "label": "Portfolio analytics",
                "why_it_matters": "Common adjacent module",
                "status": "corroborated_expansion",
                "evidence_urls": ["https://example.com/alpha"],
                "source_entity_names": ["Example Source"],
                "expansion_type": "adjacent_capability",
            },
            {
                "label": "Risk reporting",
                "why_it_matters": "Often bundled nearby",
                "status": "corroborated_expansion",
                "evidence_urls": ["https://example.com/alpha"],
                "source_entity_names": ["Example Source"],
                "expansion_type": "adjacent_capability",
            },
        ],
        "adjacent_customer_segments": [],
        "named_account_anchors": [],
        "geography_expansions": [],
    }

    report = build_expansion_report_artifact(
        source_company={"name": "Acme", "website": "https://acme.example.com"},
        expansion_brief=expansion_brief,
        source_documents=[],
        context_pack_v2={},
        confirmed_at=None,
    )

    summary_sentence = report["sections"][0]["blocks"][0]["sentences"][0]
    assert len(summary_sentence["citation_pill_ids"]) == 1


def test_derive_discovery_scope_hints_prefers_source_and_adjacent_scope():
    profile = _build_profile()
    payload = build_company_context_artifacts(profile)
    payload["expansion_brief"] = _build_expansion_payload(profile, payload)["expansion_brief"]

    scope_hints = derive_discovery_scope_hints(payload, profile)

    assert "Portfolio analytics" in scope_hints["source_capabilities"]
    assert "Northwind Capital" in scope_hints["named_account_anchors"]
    assert scope_hints["confirmed"] is False


def test_build_company_context_artifacts_builds_expansion_inputs_from_non_buyer_sites():
    profile = _build_profile()
    profile.comparator_seed_urls = []
    profile.context_pack_json["sites"].append(
        {
            "url": "https://comp-one.example.com",
            "company_name": "Comp One",
            "summary": "Fund operations workflow software for institutional investors.",
            "signals": [
                {
                    "type": "capability",
                    "value": "Fund operations workflow",
                    "source_url": "https://comp-one.example.com/platform",
                }
            ],
            "customer_evidence": [],
            "pages": [],
        }
    )

    payload = build_company_context_artifacts(profile)

    assert len(payload["expansion_inputs"]) == 1
    expansion_input = payload["expansion_inputs"][0]
    assert expansion_input["name"] == "Comp One"
    assert expansion_input["website"] == "https://comp-one.example.com"


def test_build_company_context_artifacts_falls_back_when_seed_domains_do_not_match_sites():
    profile = _build_profile()
    profile.comparator_seed_urls = ["https://stale.example.com"]
    profile.context_pack_json["sites"].append(
        {
            "url": "https://comp-one.example.com",
            "company_name": "Comp One",
            "summary": "Fund operations workflow software for institutional investors.",
            "signals": [
                {
                    "type": "capability",
                    "value": "Fund operations workflow",
                    "source_url": "https://comp-one.example.com/platform",
                }
            ],
            "customer_evidence": [],
            "pages": [],
        }
    )

    payload = build_company_context_artifacts(profile)

    assert len(payload["expansion_inputs"]) == 1
    expansion_input = payload["expansion_inputs"][0]
    assert expansion_input["name"] == "Comp One"
    assert expansion_input["website"] == "https://comp-one.example.com"


def test_build_company_context_artifacts_derives_expansion_brief_from_comparators():
    profile = _build_profile()
    profile.geo_scope = {"region": "EU+UK", "include_countries": ["Belgium"], "exclude_countries": []}
    profile.context_pack_json["sites"].append(
        {
            "url": "https://comp-one.example.com",
            "company_name": "Comp One",
            "summary": "Client reporting and proxy voting software for private banks and asset managers.",
            "signals": [
                {
                    "type": "capability",
                    "value": "Client reporting",
                    "source_url": "https://comp-one.example.com/reporting",
                },
                {
                    "type": "capability",
                    "value": "Proxy voting",
                    "source_url": "https://comp-one.example.com/voting",
                },
                {
                    "type": "customer",
                    "value": "Private bank",
                    "source_url": "https://comp-one.example.com/clients",
                },
                {
                    "type": "customer",
                    "value": "Fund administrator",
                    "source_url": "https://comp-one.example.com/clients",
                },
            ],
            "customer_evidence": [],
            "pages": [],
        }
    )

    payload = build_company_context_artifacts(profile)
    expansion = _build_expansion_payload(profile, payload)["expansion_brief"]

    assert any(item["label"] == "Client reporting" for item in expansion["adjacent_capabilities"])
    assert any(item["label"] == "Fund administrator" for item in expansion["adjacent_customer_segments"])
    assert any(item["label"] == "Belgium" for item in expansion["geography_expansions"])


def test_build_company_context_artifacts_uses_domain_brand_for_generic_comparator_titles():
    profile = _build_profile()
    profile.comparator_seed_urls = ["https://www.zaggo.fr/"]
    profile.context_pack_json["sites"].append(
        {
            "url": "https://www.zaggo.fr/",
            "company_name": "Gestion des remplacements en urgence",
            "summary": "L'application Zaggo permet une gestion des remplacements urgents et imprévus.",
            "signals": [
                {
                    "type": "capability",
                    "value": "Gestion des remplacements en urgence",
                    "source_url": "https://www.zaggo.fr/",
                }
            ],
            "customer_evidence": [],
            "pages": [],
        }
    )

    payload = build_company_context_artifacts(profile)

    assert payload["expansion_inputs"]
    assert payload["expansion_inputs"][0]["name"] == "Zaggo"


def test_build_company_context_artifacts_derives_comparator_capabilities_from_page_titles():
    profile = _build_profile()
    profile.comparator_seed_urls = ["https://www.zaggo.fr/"]
    profile.context_pack_json["sites"].append(
        {
            "url": "https://www.zaggo.fr/",
            "company_name": "Gestion des remplacements en urgence",
            "summary": "",
            "signals": [],
            "customer_evidence": [],
            "pages": [
                {
                    "url": "https://www.zaggo.fr/",
                    "title": "Gestion des remplacements en urgence - Santé et médico-social",
                    "signals": [],
                    "blocks": [],
                },
                {
                    "url": "https://www.zaggo.fr/etablissements-sante",
                    "title": "Solution de recrutement de personnel médical en remplacement",
                    "signals": [],
                    "blocks": [],
                },
                {
                    "url": "https://www.zaggo.fr/mentions-legales",
                    "title": "Mentions légales - Zaggo.fr",
                    "signals": [],
                    "blocks": [],
                },
            ],
        }
    )

    payload = build_company_context_artifacts(profile)
    expansion_input = payload["expansion_inputs"][0]
    capability_labels = [
        node["phrase"]
        for node in expansion_input["taxonomy_nodes"]
        if node.get("layer") == "capability"
    ]

    assert "Gestion des remplacements en urgence" in capability_labels
    assert "Recrutement de personnel médical en remplacement" in capability_labels
    assert all("Mentions légales" not in label for label in capability_labels)


def test_build_company_context_artifacts_derives_mediflash_style_staffing_capability_from_title():
    profile = _build_profile()
    profile.comparator_seed_urls = ["https://mediflash.fr/"]
    profile.context_pack_json["sites"].append(
        {
            "url": "https://mediflash.fr/",
            "company_name": "Mediflash",
            "summary": "",
            "signals": [],
            "customer_evidence": [],
            "pages": [
                {
                    "url": "https://mediflash.fr/",
                    "title": "Missions de renfort soignant - Mediflash",
                    "signals": [],
                    "blocks": [],
                },
                {
                    "url": "https://mediflash.fr/politique-de-confidentialite",
                    "title": "Politique de confidentialité - Mediflash",
                    "signals": [],
                    "blocks": [],
                },
            ],
        }
    )

    payload = build_company_context_artifacts(profile)
    expansion_input = payload["expansion_inputs"][0]
    capability_labels = [
        node["phrase"]
        for node in expansion_input["taxonomy_nodes"]
        if node.get("layer") == "capability"
    ]

    assert "Renfort soignant" in capability_labels
    assert all("Politique" not in label for label in capability_labels)


def test_build_company_context_artifacts_uses_model_backed_expansion_brief(monkeypatch):
    profile = _build_profile()
    profile.geo_scope = {"region": "EU+UK", "include_countries": ["Belgium"], "exclude_countries": []}

    monkeypatch.setattr(
        "app.services.company_context._reason_sourcing_brief",
        lambda **kwargs: {
            **kwargs["fallback_brief"],
            "reasoning_status": "success",
            "reasoning_warning": None,
            "reasoning_provider": "test",
            "reasoning_model": "stub",
        },
    )
    monkeypatch.setattr("app.services.company_context.get_settings", lambda: type("S", (), {
        "gemini_api_key": "x",
        "openai_api_key": "",
        "anthropic_api_key": "",
    })())

    def _fake_run_stage(_self, request):
        if request.stage == LLMStage.expansion_brief_reasoning:
            return LLMResponse(
                text=json.dumps(
                    {
                        "adjacent_capabilities": [
                            {
                                "label": "Voting rights / proxy voting",
                                "why_it_matters": "Adjacent governance workflow around securities operations.",
                                "evidence_urls": ["https://example.com/voting"],
                                "source_entity_names": ["Comparator A"],
                                "market_importance": "low",
                                "operational_centrality": "peripheral",
                                "confidence": 0.58,
                            }
                        ],
                        "adjacent_customer_segments": [],
                        "named_account_anchors": [],
                        "geography_expansions": [],
                    }
                ),
                provider="gemini",
                model="gemini-2.0-flash",
            )
        if request.stage == LLMStage.structured_normalization:
            return LLMResponse(
                text=json.dumps(
                    {
                        "reasoning_status": "success",
                        "reasoning_warning": None,
                        "adjacent_capabilities": [
                            {
                                "id": "expansion_voting",
                                "label": "Voting rights / proxy voting",
                                "expansion_type": "adjacent_capability",
                                "status": "hypothesis",
                                "confidence": 0.58,
                                "why_it_matters": "Adjacent governance workflow around securities operations.",
                                "evidence_urls": ["https://example.com/voting"],
                                "supporting_node_ids": [],
                                "source_entity_names": ["Comparator A"],
                                "market_importance": "low",
                                "operational_centrality": "peripheral",
                                "priority_tier": "edge_case",
                            }
                        ],
                        "adjacent_customer_segments": [],
                        "named_account_anchors": [],
                        "geography_expansions": [],
                    }
                ),
                provider="openai",
                model="gpt-4.1-mini",
            )
        raise AssertionError(f"Unexpected stage: {request.stage}")

    monkeypatch.setattr("app.services.company_context.LLMOrchestrator.run_stage", _fake_run_stage)

    payload = build_company_context_artifacts(profile)
    expansion = _build_expansion_payload(profile, payload)["expansion_brief"]

    assert expansion["reasoning_status"] == "success"
    assert expansion["adjacent_capabilities"][0]["label"] == "Voting rights / proxy voting"
    assert expansion["adjacent_capabilities"][0]["market_importance"] == "low"
    assert expansion["adjacent_capabilities"][0]["operational_centrality"] == "peripheral"
    assert expansion["adjacent_capabilities"][0]["priority_tier"] == "edge_case"


def test_build_company_context_artifacts_prefers_comparator_capabilities_over_source_self_paraphrases(monkeypatch):
    profile = _build_profile()
    profile.context_pack_json["sites"].append(
        {
            "url": "https://comp-one.example.com",
            "company_name": "Comp One",
            "summary": "Client reporting and proxy voting software for private banks.",
            "signals": [
                {
                    "type": "capability",
                    "value": "Proxy voting",
                    "source_url": "https://comp-one.example.com/voting",
                }
            ],
            "customer_evidence": [],
            "pages": [],
        }
    )

    monkeypatch.setattr(
        "app.services.company_context._reason_sourcing_brief",
        lambda **kwargs: {
            **kwargs["fallback_brief"],
            "reasoning_status": "success",
            "reasoning_warning": None,
            "reasoning_provider": "test",
            "reasoning_model": "stub",
        },
    )
    monkeypatch.setattr("app.services.company_context.get_settings", lambda: type("S", (), {
        "gemini_api_key": "x",
        "openai_api_key": "",
        "anthropic_api_key": "",
    })())

    def _fake_run_stage(_self, request):
        if request.stage == LLMStage.expansion_brief_reasoning:
            return LLMResponse(
                text=json.dumps(
                    {
                        "adjacent_capabilities": [
                            {
                                "label": "Portfolio analytics automation",
                                "why_it_matters": "Restates the source company capability.",
                                "evidence_urls": ["https://acme.example.com/platform"],
                                "source_entity_names": ["Acme"],
                                "status": "source_grounded",
                            }
                        ],
                        "adjacent_customer_segments": [],
                        "named_account_anchors": [],
                        "geography_expansions": [],
                    }
                ),
                provider="gemini",
                model="gemini-2.0-flash",
            )
        if request.stage == LLMStage.structured_normalization:
            return LLMResponse(
                text=json.dumps(
                    {
                        "reasoning_status": "success",
                        "reasoning_warning": None,
                        "adjacent_capabilities": [
                            {
                                "id": "expansion_source_self",
                                "label": "Portfolio analytics automation",
                                "expansion_type": "adjacent_capability",
                                "status": "source_grounded",
                                "confidence": 0.7,
                                "why_it_matters": "Restates the source company capability.",
                                "evidence_urls": ["https://acme.example.com/platform"],
                                "supporting_node_ids": [],
                                "source_entity_names": ["Acme"],
                                "market_importance": "medium",
                                "operational_centrality": "core",
                                "priority_tier": "meaningful_adjacent",
                            }
                        ],
                        "adjacent_customer_segments": [],
                        "named_account_anchors": [],
                        "geography_expansions": [],
                    }
                ),
                provider="openai",
                model="gpt-4.1-mini",
            )
        raise AssertionError(f"Unexpected stage: {request.stage}")

    monkeypatch.setattr("app.services.company_context.LLMOrchestrator.run_stage", _fake_run_stage)

    payload = build_company_context_artifacts(profile)
    expansion = _build_expansion_payload(profile, payload)["expansion_brief"]

    capability_labels = [item["label"] for item in expansion["adjacent_capabilities"]]
    assert "Proxy voting" in capability_labels
    assert "Portfolio analytics automation" not in capability_labels


def test_build_company_context_artifacts_keeps_fallback_customer_segments_when_model_omits_them(monkeypatch):
    profile = _build_profile()
    profile.context_pack_json["sites"].append(
        {
            "url": "https://comp-one.example.com",
            "company_name": "Comp One",
            "summary": "Solutions de staffing pour établissements de santé.",
            "signals": [
                {
                    "type": "customer",
                    "value": "Établissements de santé",
                    "source_url": "https://comp-one.example.com/clients",
                }
            ],
            "customer_evidence": [],
            "pages": [],
        }
    )

    monkeypatch.setattr(
        "app.services.company_context._reason_sourcing_brief",
        lambda **kwargs: {
            **kwargs["fallback_brief"],
            "reasoning_status": "success",
            "reasoning_warning": None,
            "reasoning_provider": "test",
            "reasoning_model": "stub",
        },
    )
    monkeypatch.setattr("app.services.company_context.get_settings", lambda: type("S", (), {
        "gemini_api_key": "x",
        "openai_api_key": "",
        "anthropic_api_key": "",
    })())

    def _fake_run_stage(_self, request):
        if request.stage == LLMStage.expansion_brief_reasoning:
            return LLMResponse(
                text=json.dumps(
                    {
                        "adjacent_capabilities": [],
                        "adjacent_customer_segments": [],
                        "named_account_anchors": [],
                        "geography_expansions": [],
                    }
                ),
                provider="gemini",
                model="gemini-2.0-flash",
            )
        raise AssertionError(f"Unexpected stage: {request.stage}")

    monkeypatch.setattr("app.services.company_context.LLMOrchestrator.run_stage", _fake_run_stage)

    payload = build_company_context_artifacts(profile)
    expansion = _build_expansion_payload(profile, payload)["expansion_brief"]

    assert any(item["label"] == "Healthcare provider" for item in expansion["adjacent_customer_segments"])


def test_build_expansion_artifacts_filters_noisy_named_account_anchors():
    profile = _build_profile()
    expansion = build_expansion_artifacts(
        profile,
        sourcing_brief={
            "source_company": {"name": "Hublo", "website": "https://hublo.com"},
            "capability_nodes": [{"phrase": "Shift replacement"}],
            "customer_nodes": [{"phrase": "Hospital"}],
            "named_customer_proof": [
                {"name": "Adrien Beata", "source_url": "https://hublo.com/fr/customers"},
                {"name": "Image représentant un personnage barbu", "source_url": "https://hublo.com/fr/customers"},
                {"name": "AP-HP", "source_url": "https://hublo.com/fr/customers"},
            ],
        },
        taxonomy_nodes=[],
    )["expansion_brief"]

    labels = [item["label"] for item in expansion["named_account_anchors"]]
    assert "AP-HP" in labels
    assert "Adrien Beata" not in labels
    assert all("Image représentant" not in label for label in labels)


def test_build_company_context_artifacts_filters_person_like_named_customer_proof():
    profile = _build_profile()
    profile.context_pack_json["sites"][0]["pages"] = [
        {
            "url": "https://acme.example.com/customers",
            "title": "Customers",
            "signals": [],
            "customer_evidence": [
                {"name": "Adrien Beata", "source_url": "https://acme.example.com/customers", "evidence_type": "logo_alt"},
                {
                    "name": "Image représentant un personnage barbu",
                    "source_url": "https://acme.example.com/customers",
                    "evidence_type": "logo_alt",
                },
                {
                    "name": "Le schéma Hublo et agences",
                    "source_url": "https://acme.example.com/customers",
                    "evidence_type": "logo_alt",
                },
                {"name": "Northwind Capital", "source_url": "https://acme.example.com/customers", "evidence_type": "logo_alt"},
            ],
        }
    ]
    profile.context_pack_json = build_context_pack_v2(profile.context_pack_json)

    payload = build_company_context_artifacts(profile)
    labels = [item["name"] for item in payload["sourcing_brief"]["named_customer_proof"]]

    assert "Northwind Capital" in labels
    assert "Adrien Beata" not in labels
    assert all("Image représentant" not in label for label in labels)
    assert "Le schéma Hublo et agences" not in labels


def test_build_expansion_artifacts_filters_noisy_adjacent_capabilities():
    profile = _build_profile()
    noisy_input = {
        "name": "Comp Two",
        "website": "https://comp-two.example.com",
        "taxonomy_nodes": [
            {"layer": "capability", "phrase": "Politique de protection des données"},
            {
                "layer": "capability",
                "phrase": "Créer votre vivier de remplaçants",
                "source_url": "https://comp-two.example.com/replacement-pool",
            },
        ],
        "named_customer_proof": [],
        "partner_integration_proof": [],
    }

    from app.services.company_context import _build_deterministic_expansion_brief

    deterministic = _build_deterministic_expansion_brief(
        profile=profile,
        sourcing_brief={
            "source_company": {"name": "Hublo", "website": "https://hublo.com"},
            "capability_nodes": [{"phrase": "Shift replacement"}],
            "customer_nodes": [{"phrase": "Hospital"}],
            "named_customer_proof": [],
        },
        taxonomy_nodes=[],
        expansion_inputs=[noisy_input],
    )

    labels = [item["label"] for item in deterministic["adjacent_capabilities"]]
    assert "Créer votre vivier de remplaçants" in labels
    assert "Politique de protection des données" not in labels
    evidence_urls = {
        item["label"]: item.get("evidence_urls") or []
        for item in deterministic["adjacent_capabilities"]
    }
    assert evidence_urls["Créer votre vivier de remplaçants"] == ["https://comp-two.example.com/replacement-pool"]


def test_normalize_expansion_brief_filters_noisy_model_outputs():
    normalized = normalize_expansion_brief(
        {
            "adjacent_capabilities": [
                {"label": "Data protection and security policies"},
                {"label": "Regulatory compliance assurance"},
                {"label": "Workforce replacement pool management"},
            ],
            "adjacent_customer_segments": [
                {"label": "Nurses and paramedical professionals"},
                {"label": "Hospitals"},
            ],
            "named_account_anchors": [
                {"label": "Photo équipe BDE Lille"},
                {"label": "Northwind Capital"},
                {
                    "label": "Rothschild & Co",
                    "why_it_matters": "Named investor and portfolio partner useful for M&A anchoring.",
                    "evidence_urls": ["https://rothschildandco.com/en/five-arrows/corporate-private-equity/portfolio/hublo"],
                },
            ],
        }
    )

    capability_labels = [item["label"] for item in normalized["adjacent_capabilities"]]
    segment_labels = [item["label"] for item in normalized["adjacent_customer_segments"]]
    named_account_labels = [item["label"] for item in normalized["named_account_anchors"]]

    assert "Workforce replacement pool management" in capability_labels
    assert "Data protection and security policies" not in capability_labels
    assert "Regulatory compliance assurance" not in capability_labels
    assert "Hospitals" in segment_labels
    assert "Nurses and paramedical professionals" not in segment_labels
    assert "Northwind Capital" in named_account_labels
    assert "Photo équipe BDE Lille" not in named_account_labels
    assert "Rothschild & Co" not in named_account_labels


def test_normalize_expansion_brief_filters_consulting_style_capabilities_and_keeps_healthcare_provider():
    normalized = normalize_expansion_brief(
        {
            "adjacent_capabilities": [
                {"label": "Des consultants du secteur RH et recrutement formation"},
                {"label": "Management de la sécurité"},
                {"label": "Reportings des remplacement effectués"},
                {"label": "Candidatdésigne une personne libre de tout engagement inscrite"},
                {"label": "Date de la notification"},
                {"label": "Des professionnels des secteurs conseil juridique"},
                {"label": "Urgent replacement management"},
                {"label": "Client reporting"},
            ],
            "adjacent_customer_segments": [
                {"label": "Établissements de santé"},
            ],
        }
    )

    capability_labels = [item["label"] for item in normalized["adjacent_capabilities"]]
    segment_labels = [item["label"] for item in normalized["adjacent_customer_segments"]]

    assert "Urgent replacement management" in capability_labels
    assert "Des consultants du secteur RH et recrutement formation" not in capability_labels
    assert "Management de la sécurité" not in capability_labels
    assert "Reportings des remplacement effectués" not in capability_labels
    assert "Candidatdésigne une personne libre de tout engagement inscrite" not in capability_labels
    assert "Date de la notification" not in capability_labels
    assert "Des professionnels des secteurs conseil juridique" not in capability_labels
    assert "Client reporting" in capability_labels
    assert "Établissements de santé" in segment_labels


def test_merge_reasoned_expansion_brief_unions_duplicate_capability_evidence_urls():
    fallback_brief = normalize_expansion_brief(
        {
            "adjacent_capabilities": [
                {
                    "label": "Recrutement de personnel médical en remplacement",
                    "evidence_urls": ["https://zaggo.fr/etablissements-sante"],
                    "source_entity_names": ["Zaggo"],
                }
            ]
        }
    )

    merged = _merge_reasoned_expansion_brief(
        response_text=json.dumps(
            {
                "adjacent_capabilities": [
                    {
                        "label": "Recrutement de personnel médical en remplacement",
                        "evidence_urls": ["https://zaggo.fr/"],
                        "source_entity_names": ["Zaggo"],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        fallback_brief=fallback_brief,
        taxonomy_nodes=[],
        source_company={"name": "Hublo", "website": "https://hublo.com"},
        comparator_domains={"zaggo.fr"},
    )

    capability = merged["adjacent_capabilities"][0]
    assert capability["label"] == "Recrutement de personnel médical en remplacement"
    assert capability["evidence_urls"] == [
        "https://zaggo.fr/etablissements-sante",
        "https://zaggo.fr/",
    ]


def test_report_source_label_distinguishes_deeper_comparator_pages():
    assert _report_source_label(
        "https://zaggo.fr/",
        publisher="Zaggo",
    ) == "Zaggo"
    assert _report_source_label(
        "https://zaggo.fr/etablissements-sante",
        publisher="Zaggo",
    ) == "Zaggo / etablissements-sante"


def test_build_sourcing_report_labels_site_summary_sources_with_domain_brand():
    report = build_sourcing_report_artifact(
        sourcing_brief={
            "source_company": {"name": "Hublo", "website": "https://hublo.com"},
            "source_summary": "Hublo summary",
            "customer_nodes": [{"phrase": "Healthcare provider", "evidence_ids": ["ev_zaggo"]}],
            "workflow_nodes": [],
            "capability_nodes": [],
            "delivery_or_integration_nodes": [],
            "named_customer_proof": [],
            "partner_integration_proof": [],
            "secondary_evidence_proof": [],
        },
        source_documents=[],
        context_pack_v2={
            "evidence_items": [
                {
                    "id": "ev_zaggo",
                    "url": "https://zaggo.fr/",
                    "page_type": "site_summary",
                    "kind": "site_signal:capability",
                }
            ]
        },
        confirmed_at=None,
    )

    source_labels = [item["label"] for item in report["sources"]]
    assert "Zaggo" in source_labels


def test_build_company_context_artifacts_filters_policy_capabilities_from_sourcing():
    profile = _build_profile()
    profile.context_pack_json["sites"][0]["pages"] = [
        {
            "url": "https://acme.example.com/platform",
            "title": "Platform",
            "signals": [
                {"type": "capability", "value": "Portfolio analytics", "source_url": "https://acme.example.com/platform"},
                {
                    "type": "capability",
                    "value": "Politique de protection des données",
                    "source_url": "https://acme.example.com/platform",
                },
                {
                    "type": "capability",
                    "value": "Vous êtes une agence d’intérim ?",
                    "source_url": "https://acme.example.com/platform",
                },
                {
                    "type": "capability",
                    "value": "Centres hospitaliers",
                    "source_url": "https://acme.example.com/platform",
                },
                {
                    "type": "capability",
                    "value": "À propos de Hublo",
                    "source_url": "https://acme.example.com/platform",
                },
            ],
            "customer_evidence": [],
        }
    ]

    payload = build_company_context_artifacts(profile)
    capability_labels = [item["phrase"] for item in payload["sourcing_brief"]["capability_nodes"]]

    assert "Portfolio analytics" in capability_labels
    assert "Politique de protection des données" not in capability_labels
    assert "Vous êtes une agence d’intérim ?" not in capability_labels
    assert "Centres hospitaliers" not in capability_labels
    assert "À propos de Hublo" not in capability_labels


def test_scope_review_decisions_compile_scope_back_into_discovery_scope_hints():
    profile = _build_profile()
    profile.geo_scope = {"region": "EU+UK", "include_countries": ["Belgium"], "exclude_countries": []}
    profile.context_pack_json["sites"].append(
        {
            "url": "https://comp-one.example.com",
            "company_name": "Comp One",
            "summary": "Client reporting and proxy voting software for private banks and fund administrators.",
            "signals": [
                {
                    "type": "capability",
                    "value": "Proxy voting",
                    "source_url": "https://comp-one.example.com/voting",
                },
                {
                    "type": "customer",
                    "value": "Fund administrator",
                    "source_url": "https://comp-one.example.com/clients",
                },
            ],
            "customer_evidence": [],
            "pages": [],
        }
    )

    payload = build_company_context_artifacts(profile)
    payload["expansion_brief"] = _build_expansion_payload(profile, payload)["expansion_brief"]
    scope_review = derive_scope_review_payload(payload, profile)
    source_capability = next(item for item in scope_review["source_capabilities"] if item["label"] == "Portfolio analytics")
    adjacent_capability = next(
        item for item in scope_review["adjacent_capabilities"] if item["label"] == "Proxy voting"
    )

    adjusted = apply_scope_review_decisions(
        payload,
        [
            {"id": source_capability["id"], "status": "user_removed"},
            {"id": adjacent_capability["id"], "status": "user_kept"},
        ],
    )

    scope_hints = derive_discovery_scope_hints(adjusted, profile)
    adjusted_node = next(node for node in adjusted["taxonomy_nodes"] if node["id"] == source_capability["id"])

    assert adjusted_node["scope_status"] == "removed"
    assert "Proxy voting" in scope_hints["adjacent_capabilities"]


def test_scope_review_payload_includes_evidence_urls_for_source_nodes():
    profile = _build_profile()
    payload = build_company_context_artifacts(profile)

    scope_review = derive_scope_review_payload(payload, profile)
    source_capability = next(item for item in scope_review["source_capabilities"] if item["label"] == "Portfolio analytics")

    assert source_capability["evidence_ids"]
    assert source_capability["evidence_urls"]
    assert source_capability["evidence_urls"][0].startswith("https://acme.example.com")


def test_capability_phrase_from_text_prefers_valid_segment_from_split_title():
    assert (
        _capability_phrase_from_text("Créer votre vivier de remplaçants - Simplifier votre")
        == "Créer votre vivier de remplaçants"
    )


def test_capability_phrase_from_text_drops_brand_only_split_titles():
    assert _capability_phrase_from_text("Hublo - Mstaff") == ""


def test_build_company_context_artifacts_ignores_comparator_context_for_empty_buyer_site():
    profile = CompanyProfile(
        workspace_id=3,
        buyer_company_url="https://4tpm.fr/",
        comparator_seed_urls=["https://cwan.com/"],
        supporting_evidence_urls=[],
        comparator_seed_summaries={
            "https://cwan.com/": (
                "CWAN serves institutional investors with cloud deployment, integrations, "
                "and implementation services."
            )
        },
        geo_scope={"region": "EU+UK", "include_countries": [], "exclude_countries": []},
        context_pack_markdown="# 4TPM\n\n**Website:** https://4tpm.fr\n\n---\n\n# CWAN\n\nCloud-native SaaS for asset managers.",
        context_pack_json={
            "sites": [
                {
                    "url": "https://4tpm.fr/",
                    "company_name": "4TPM",
                    "summary": "",
                    "signals": [],
                    "customer_evidence": [],
                    "pages": [
                        {
                            "url": "https://4tpm.fr/",
                            "title": "4TPM - Plateforme Wealth Management",
                            "blocks": [],
                            "signals": [],
                            "customer_evidence": [],
                            "raw_content": "",
                        }
                    ],
                },
                {
                    "url": "https://cwan.com/",
                    "company_name": "CWAN",
                    "summary": "Cloud-native SaaS for institutional investors.",
                    "signals": [
                        {
                            "type": "capability",
                            "value": "Portfolio analytics",
                            "source_url": "https://cwan.com/platform",
                        }
                    ],
                    "customer_evidence": [
                        {
                            "name": "Asset managers",
                            "source_url": "https://cwan.com/customers",
                        }
                    ],
                    "pages": [],
                },
            ]
        },
        product_pages_found=0,
    )

    payload = build_company_context_artifacts(profile)

    assert payload["buyer_evidence"]["status"] == "insufficient"
    assert payload["buyer_evidence"]["used_for_inference"] is False
    assert payload["sourcing_brief"]["source_summary"]
    assert payload["sourcing_brief"]["customer_nodes"] == []


def test_build_context_pack_v2_keeps_high_signal_job_pages():
    context_pack = build_context_pack_v2(
        {
            "generated_at": "2026-03-12T00:00:00Z",
            "sites": [
                {
                    "url": "https://hublo.com",
                    "company_name": "Hublo",
                    "pages": [
                        {
                            "url": "https://careers.hublo.com/jobs/7331974-head-of-data-ai-f-h-n",
                            "title": "Head of Data & AI",
                            "page_type": "careers",
                            "blocks": [{"type": "heading", "content": "Data platform for healthcare staffing"}],
                            "signals": [],
                            "customer_evidence": [],
                            "raw_content": "Work closely with product, customers, operations, AI, and scheduling workflows.",
                        },
                        {
                            "url": "https://careers.hublo.com/jobs/talent-partner",
                            "title": "Talent Partner",
                            "page_type": "careers",
                            "blocks": [],
                            "signals": [],
                            "customer_evidence": [],
                            "raw_content": "General recruiting process and interviews.",
                        },
                    ],
                }
            ],
        }
    )

    selected_pages = context_pack["sites"][0]["selected_pages"]
    selected_urls = {page["url"] for page in selected_pages}
    assert "https://careers.hublo.com/jobs/7331974-head-of-data-ai-f-h-n" in selected_urls
    assert "https://careers.hublo.com/jobs/talent-partner" not in selected_urls


def test_build_context_pack_v2_filters_noisy_customer_evidence_and_uses_raw_text_for_phrases():
    context_pack = build_context_pack_v2(
        {
            "generated_at": "2026-03-12T00:00:00Z",
            "sites": [
                {
                    "url": "https://wealth.example.com",
                    "company_name": "WealthCo",
                    "summary": "Client lifecycle management software for private banks and wealth managers.",
                    "customer_evidence": [
                        {
                            "name": "Why first impressions matter: Rethinking onboarding in wealth management",
                            "source_url": "https://wealth.example.com/insights",
                            "evidence_type": "logo_alt",
                            "context": "Recent insights",
                        },
                        {
                            "name": "BNP Paribas",
                            "source_url": "https://wealth.example.com/customers",
                            "evidence_type": "logo_alt",
                            "context": "Trusted by leading private banks",
                        },
                    ],
                    "pages": [
                        {
                            "url": "https://wealth.example.com/solutions/clm",
                            "title": "Client lifecycle management for private banks",
                            "page_type": "solutions",
                            "blocks": [],
                            "signals": [],
                            "customer_evidence": [],
                            "raw_content": (
                                "Our client lifecycle management platform helps private banks streamline "
                                "onboarding and portfolio reporting workflows."
                            ),
                        }
                    ],
                }
            ],
        }
    )

    site = context_pack["sites"][0]
    assert [item["name"] for item in site["named_customers"]] == ["BNP Paribas"]
    assert "private bank" in [phrase.lower() for phrase in context_pack["extracted_raw_phrases"]]
    assert "portfolio reporting" in [phrase.lower() for phrase in context_pack["extracted_raw_phrases"]]


def test_build_company_context_artifacts_builds_taxonomy_from_spa_style_phrases():
    profile = CompanyProfile(
        workspace_id=4,
        buyer_company_url="https://4tpm.fr/",
        comparator_seed_urls=[],
        supporting_evidence_urls=[],
        comparator_seed_summaries={},
        geo_scope={"region": "EU+UK", "include_countries": [], "exclude_countries": []},
        context_pack_json={
            "version": "v2",
            "generated_at": "2026-03-12T00:00:00Z",
            "sites": [
                {
                    "url": "https://4tpm.fr/",
                    "company_name": "4TPM",
                    "website": "https://4tpm.fr/",
                    "selected_pages": [],
                }
            ],
                "evidence_items": [
                    {"id": "e1", "kind": "page_signal:customer_archetype", "text": "Banques privées"},
                    {"id": "e1b", "kind": "page_signal:workflow", "text": "Front office"},
                    {"id": "e2", "kind": "page_signal:workflow", "text": "Front office titres"},
                    {"id": "e2b", "kind": "page_signal:workflow", "text": "Operations"},
                    {"id": "e2c", "kind": "page_signal:workflow", "text": "Infrastructure"},
                    {"id": "e2d", "kind": "page_signal:workflow", "text": "4TPM PATIO"},
                    {"id": "e3", "kind": "page_signal:service", "text": "Documentation API"},
                    {"id": "e3b", "kind": "page_signal:service", "text": "APIs REST"},
                    {"id": "e3c", "kind": "page_signal:integration", "text": "BPCE"},
                    {"id": "e4", "kind": "page_customer:bundle_logo_manifest", "text": "Procapital"},
                ],
            "named_customers": [
                {"name": "Allianz Bank", "evidence_id": "cust1", "context": "Trusted by leading banques privées"}
            ],
            "integrations": [],
            "partners": [],
                "extracted_raw_phrases": [
                    "Banques privées",
                    "Bourse en ligne",
                    "Front office titres",
                    "Front office",
                    "Operations",
                    "Infrastructure",
                    "4TPM PATIO",
                    "Back office titres",
                    "Documentation API",
                    "APIs REST",
                    "4TPM - Plateforme Wealth Management",
                    "BPCE",
                    "Procapital",
                ],
                "crawl_coverage": {"total_sites": 1, "total_pages": 4},
            },
        product_pages_found=4,
    )

    payload = build_company_context_artifacts(profile)
    taxonomy_by_layer = {}
    for node in payload["taxonomy_nodes"]:
        taxonomy_by_layer.setdefault(node["layer"], []).append(node["phrase"])

    assert "Private bank" in taxonomy_by_layer["customer_archetype"]
    assert "Online brokerage" in taxonomy_by_layer["customer_archetype"]
    assert "Front office titres" in taxonomy_by_layer["workflow"]
    assert "Front office" not in taxonomy_by_layer["workflow"]
    assert "Operations" not in taxonomy_by_layer["workflow"]
    assert "Infrastructure" not in taxonomy_by_layer["workflow"]
    assert "4TPM PATIO" not in taxonomy_by_layer["workflow"]
    assert "API documentation" in taxonomy_by_layer["delivery_or_integration"]
    assert "REST API" in taxonomy_by_layer["delivery_or_integration"]
    assert "Infrastructure" in taxonomy_by_layer["delivery_or_integration"]
    assert "BPCE" not in taxonomy_by_layer["delivery_or_integration"]
    assert "Wealth management platform" in taxonomy_by_layer["capability"]
    assert "Procapital" not in taxonomy_by_layer.get("capability", [])
    assert "Bank" not in taxonomy_by_layer["customer_archetype"]


def test_build_company_context_artifacts_scopes_market_map_to_buyer_site():
    profile = CompanyProfile(
        workspace_id=9,
        buyer_company_url="https://4tpm.fr/",
                comparator_seed_urls=["https://wealth-dynamix.com/"],
        supporting_evidence_urls=[],
        comparator_seed_summaries={},
        geo_scope={"region": "EU+UK", "include_countries": [], "exclude_countries": []},
        context_pack_json={
            "version": "v2",
            "generated_at": "2026-03-12T00:00:00Z",
            "sites": [
                {
                    "url": "https://4tpm.fr/",
                    "company_name": "4TPM",
                    "website": "https://4tpm.fr/",
                    "selected_pages": [],
                    "evidence_items": [
                        {"id": "s1", "kind": "page_signal:customer_archetype", "text": "Banques privées"},
                        {"id": "s2", "kind": "page_signal:workflow", "text": "Front office titres"},
                        {"id": "s3", "kind": "page_signal:service", "text": "Documentation API"},
                    ],
                    "named_customers": [
                        {"name": "Allianz Bank", "evidence_id": "cust1", "context": "Trusted by leading banques privées"}
                    ],
                    "integrations": [{"name": "BPCE", "evidence_id": "int1"}],
                    "partners": [{"name": "BPCE", "evidence_id": "int1"}],
                    "extracted_raw_phrases": [
                        "Banques privées",
                        "Front office titres",
                        "Documentation API",
                        "4TPM - Plateforme Wealth Management",
                    ],
                    "crawl_coverage": {"total_pages": 4, "page_type_counts": {"other": 4}, "pages_with_signals": 4, "pages_with_customer_evidence": 1, "career_pages_selected": 0},
                },
                {
                    "url": "https://wealth-dynamix.com/",
                    "company_name": "Wealth Dynamix",
                    "website": "https://wealth-dynamix.com/",
                    "selected_pages": [],
                    "evidence_items": [
                        {"id": "c1", "kind": "page_signal:workflow", "text": "Portfolio management"},
                        {"id": "c2", "kind": "page_signal:capability", "text": "Integrate data from all sources with a unified data model"},
                    ],
                    "named_customers": [{"name": "Salesforce", "evidence_id": "cust2"}],
                    "integrations": [],
                    "partners": [],
                    "extracted_raw_phrases": [
                        "Portfolio management",
                        "Integrate data from all sources with a unified data model",
                    ],
                    "crawl_coverage": {"total_pages": 10, "page_type_counts": {"product": 10}, "pages_with_signals": 10, "pages_with_customer_evidence": 1, "career_pages_selected": 0},
                },
            ],
            "evidence_items": [],
            "named_customers": [],
            "integrations": [],
            "partners": [],
            "extracted_raw_phrases": [],
            "crawl_coverage": {"total_sites": 2, "total_pages": 14},
        },
        product_pages_found=4,
    )

    payload = build_company_context_artifacts(profile)
    taxonomy_by_layer = {}
    for node in payload["taxonomy_nodes"]:
        taxonomy_by_layer.setdefault(node["layer"], []).append(node["phrase"])

    assert "Front office titres" in taxonomy_by_layer["workflow"]
    assert "Wealth management platform" in taxonomy_by_layer["capability"]
    assert "Portfolio management" not in taxonomy_by_layer["workflow"]
    assert "Integrate data from all sources with" not in taxonomy_by_layer["capability"]


def test_build_company_context_artifacts_promotes_rendered_product_features_into_cleaner_capabilities():
    profile = CompanyProfile(
        workspace_id=10,
        buyer_company_url="https://4tpm.fr/platform/front-office",
                comparator_seed_urls=[],
        supporting_evidence_urls=[],
        comparator_seed_summaries={},
        geo_scope={},
        context_pack_json={
            "sites": [
                {
                    "url": "https://4tpm.fr/platform/front-office",
                    "company_name": "4TPM",
                    "summary": "",
                    "signals": [],
                    "customer_evidence": [],
                    "pages": [
                        {
                            "url": "https://4tpm.fr/platform/front-office",
                            "title": "4TPM - Plateforme Wealth Management",
                            "page_type": "product",
                            "blocks": [
                                {"type": "paragraph", "content": "Plateforme / Front Office titres"},
                                {"type": "heading", "content": "Front office titres pour gérants et desks de trading", "level": 1},
                                {"type": "heading", "content": "Capacités clés", "level": 2},
                                {"type": "heading", "content": "Préparation et modélisation des portefeuilles", "level": 3},
                                {"type": "heading", "content": "Génération d'ordres blocs et routage full STP", "level": 3},
                                {"type": "heading", "content": "Architecture & Intégration", "level": 2},
                                {
                                    "type": "paragraph",
                                    "content": "FIX, fichiers, MQSeries, web services, APIs REST pour se connecter aux brokers et systèmes tiers.",
                                },
                                {"type": "heading", "content": "Fonctionnalités détaillées", "level": 2},
                                {
                                    "type": "list",
                                    "content": (
                                        "- Classification des clients par listes, contraintes et périmètre de gestion\n"
                                        "- Modèles de répartition d'actifs adaptés aux profils d'investissement\n"
                                        "- Le client alimente son compte espèces via un compte de dépôt appartenant à la banque"
                                    ),
                                },
                            ],
                            "signals": [],
                            "customer_evidence": [],
                            "raw_content": "Front office titres PMS OMS REST API",
                        }
                    ],
                }
            ]
        },
        product_pages_found=1,
    )

    payload = build_company_context_artifacts(profile)
    taxonomy_by_layer: dict[str, list[str]] = {}
    for node in payload["taxonomy_nodes"]:
        taxonomy_by_layer.setdefault(node["layer"], []).append(node["phrase"])
    surfaced_capabilities = [
        node["phrase"] for node in (payload["sourcing_brief"].get("capability_nodes") or [])
    ]

    assert "Préparation et modélisation des portefeuilles" in taxonomy_by_layer["capability"]
    assert "Génération d'ordres blocs et routage full STP" in taxonomy_by_layer["capability"]
    assert "Le client alimente son compte espèces" not in taxonomy_by_layer["capability"]
    assert "REST API" in taxonomy_by_layer["delivery_or_integration"]
    assert "Préparation et modélisation des portefeuilles" in surfaced_capabilities
    assert "Génération d'ordres blocs et routage full STP" in surfaced_capabilities
    assert "Le client alimente son compte espèces" not in surfaced_capabilities


def test_taxonomy_ranking_prefers_broader_capability_clusters():
    profile = CompanyProfile(
        workspace_id=16,
        buyer_company_url="https://4tpm.fr/platform/front-office",
                comparator_seed_urls=[],
        supporting_evidence_urls=[],
        comparator_seed_summaries={},
        geo_scope={},
        context_pack_json={
            "sites": [
                {
                    "url": "https://4tpm.fr/platform/front-office",
                    "company_name": "4TPM",
                    "summary": "",
                    "signals": [],
                    "customer_evidence": [],
                    "pages": [
                        {
                            "url": "https://4tpm.fr/platform/front-office",
                            "title": "4TPM - Plateforme Wealth Management",
                            "page_type": "product",
                            "blocks": [
                                {"type": "heading", "content": "Pré-trade, trading et post-trade", "level": 3},
                                {"type": "heading", "content": "Préparation, modélisation et structuration", "level": 3},
                                {"type": "heading", "content": "Portefeuilles modèles pour le stock picking", "level": 3},
                                {"type": "heading", "content": "Contrôles de sécurité sur les retraits", "level": 3},
                            ],
                            "signals": [],
                            "customer_evidence": [],
                            "raw_content": "PMS OMS trading routing compliance",
                        }
                    ],
                }
            ]
        },
        product_pages_found=1,
    )

    payload = build_company_context_artifacts(profile)
    capability_phrases = [
        node["phrase"] for node in payload["taxonomy_nodes"] if node["layer"] == "capability"
    ]

    assert capability_phrases.index("Pré-trade, trading et post-trade") < capability_phrases.index(
        "Portefeuilles modèles pour le stock picking"
    )
    assert capability_phrases.index("Préparation, modélisation et structuration") < capability_phrases.index(
        "Contrôles de sécurité sur les retraits"
    )


def test_build_company_context_artifacts_uses_market_map_reasoning_when_available(monkeypatch):
    profile = CompanyProfile(
        workspace_id=11,
        buyer_company_url="https://4tpm.fr/platform/front-office",
                comparator_seed_urls=[],
        supporting_evidence_urls=[],
        comparator_seed_summaries={},
        geo_scope={},
        context_pack_json={
            "sites": [
                {
                    "url": "https://4tpm.fr/platform/front-office",
                    "company_name": "4TPM",
                    "summary": "",
                    "signals": [],
                    "customer_evidence": [],
                    "pages": [
                        {
                            "url": "https://4tpm.fr/platform/front-office",
                            "title": "4TPM - Plateforme Wealth Management",
                            "page_type": "product",
                            "blocks": [
                                {"type": "heading", "content": "Front office titres pour gérants et desks de trading", "level": 1},
                                {"type": "heading", "content": "Préparation et modélisation des portefeuilles", "level": 3},
                                {"type": "heading", "content": "Génération d'ordres blocs et routage full STP", "level": 3},
                                {"type": "heading", "content": "Architecture & Intégration", "level": 2},
                                {"type": "paragraph", "content": "APIs REST et documentation API."},
                            ],
                            "signals": [],
                            "customer_evidence": [],
                            "raw_content": "Front office titres PMS OMS REST API",
                        }
                    ],
                }
            ]
        },
        product_pages_found=1,
    )

    class _FakeResponse:
        def __init__(self, text: str):
            self.text = text
            self.provider = "gemini"
            self.model = "gemini-2.0-flash"

    class _FakeOrchestrator:
        def run_stage(self, request):
            import json

            assert request.stage.value == "market_map_reasoning"
            prompt = request.prompt
            input_idx = prompt.index("Input:\n") + len("Input:\n")
            payload = json.loads(prompt[input_idx:])
            assert payload["prompt_version"] == "v2"
            assert payload["selection_rules"]["max_capability_nodes"] == 6
            assert "ranked_nodes_by_layer" in payload
            assert payload["ranked_nodes_by_layer"]["capability"]
            assert payload["evidence_highlights"]["top_capability_phrases"]
            node_by_phrase = {
                node["phrase"]: node["id"]
                for node in payload["taxonomy_nodes"]
            }
            return _FakeResponse(
                json.dumps(
                    {
                        "source_summary": "4TPM appears to sell front-office wealth management capabilities to private-bank and brokerage buyers.",
                        "customer_node_ids": [],
                        "workflow_node_ids": [],
                        "capability_node_ids": [
                            node_by_phrase["Préparation et modélisation des portefeuilles"],
                            node_by_phrase["Génération d'ordres blocs et routage full STP"],
                        ],
                        "delivery_or_integration_node_ids": [
                            node_by_phrase["REST API"],
                        ],
                        "active_lens_ids": [],
                        "adjacency_hypotheses": [
                            {
                                "text": "Private-bank buyers using front-office tooling may evaluate adjacent compliance and routing capabilities.",
                                "supporting_node_ids": [
                                    node_by_phrase["Préparation et modélisation des portefeuilles"],
                                    node_by_phrase["Génération d'ordres blocs et routage full STP"],
                                ],
                                "confidence": 0.77,
                            }
                        ],
                        "confidence_gaps": ["Named customer proof is still thin."],
                        "open_questions": ["Which buyer segment should the first adjacency map prioritize?"],
                    }
                )
            )

    monkeypatch.setattr("app.services.company_context.LLMOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(
        "app.services.company_context.get_settings",
        lambda: type("S", (), {"gemini_api_key": "x", "openai_api_key": "", "anthropic_api_key": ""})(),
    )

    payload = build_company_context_artifacts(profile)

    assert payload["sourcing_brief"]["source_summary"].startswith("4TPM appears to sell front-office wealth management capabilities")
    assert payload["sourcing_brief"]["reasoning_status"] == "success"
    assert payload["sourcing_brief"]["reasoning_provider"] == "gemini"
    assert payload["sourcing_brief"]["reasoning_model"] == "gemini-2.0-flash"
    assert [
        node["phrase"] for node in payload["sourcing_brief"]["capability_nodes"]
    ] == [
        "Préparation et modélisation des portefeuilles",
        "Génération d'ordres blocs et routage full STP",
    ]
    assert [
        node["phrase"] for node in payload["sourcing_brief"]["delivery_or_integration_nodes"]
    ] == ["REST API"]
    assert "Which buyer segment should the first adjacency map prioritize?" in payload["sourcing_brief"]["open_questions"]
    assert len(payload["sourcing_brief"]["open_questions"]) == 2


def test_build_company_context_artifacts_builds_hublo_style_market_map_layers():
    profile = CompanyProfile(
        workspace_id=13,
        buyer_company_url="https://www.hublo.com/en",
                comparator_seed_urls=[],
        supporting_evidence_urls=[],
        comparator_seed_summaries={},
        geo_scope={},
        context_pack_json={
            "version": "v2",
            "generated_at": "2026-03-12T00:00:00Z",
            "sites": [
                {
                    "url": "https://www.hublo.com/en",
                    "company_name": "Hublo",
                    "website": "https://www.hublo.com/en",
                    "summary": "Hublo helps hospitals and care providers manage staffing, shift replacement, and internal mobility.",
                    "pages": [
                        {
                            "url": "https://www.hublo.com/en/solutions",
                            "title": "Healthcare staffing platform",
                            "page_type": "solutions",
                            "blocks": [
                                {"type": "heading", "content": "For hospitals and care providers", "level": 1},
                                {"type": "heading", "content": "Shift replacement", "level": 2},
                                {"type": "heading", "content": "Internal mobility", "level": 2},
                                {"type": "heading", "content": "Pool management", "level": 2},
                                {"type": "heading", "content": "API documentation", "level": 2},
                            ],
                            "signals": [],
                            "customer_evidence": [],
                            "raw_content": (
                                "Hospitals use Hublo to manage staffing operations, replacement planning, "
                                "and workforce pools across departments."
                            ),
                        },
                        {
                            "url": "https://careers.hublo.com/jobs/7331974-head-of-data-ai-f-h-n",
                            "title": "Head of Data & AI",
                            "page_type": "careers",
                            "blocks": [
                                {"type": "heading", "content": "Build data products for healthcare staffing operations"}
                            ],
                            "signals": [],
                            "customer_evidence": [],
                            "raw_content": (
                                "Work with product, operations, hospitals, staffing workflows, and scheduling data."
                            ),
                        },
                    ],
                    "signals": [
                        {
                            "type": "customer_archetype",
                            "value": "Hospitals",
                            "source_url": "https://www.hublo.com/en/solutions",
                        },
                        {
                            "type": "workflow",
                            "value": "Shift replacement",
                            "source_url": "https://www.hublo.com/en/solutions",
                        },
                        {
                            "type": "workflow",
                            "value": "Internal mobility",
                            "source_url": "https://www.hublo.com/en/solutions",
                        },
                        {
                            "type": "capability",
                            "value": "Pool management",
                            "source_url": "https://www.hublo.com/en/solutions",
                        },
                        {
                            "type": "service",
                            "value": "API documentation",
                            "source_url": "https://www.hublo.com/en/solutions",
                        },
                    ],
                    "customer_evidence": [
                        {
                            "name": "AP-HP",
                            "source_url": "https://www.hublo.com/en/customers",
                            "context": "Healthcare staffing customer proof",
                            "evidence_type": "case_study",
                        }
                    ],
                }
            ],
            "named_customers": [{"name": "AP-HP", "evidence_id": "cust_hp"}],
            "integrations": [],
            "partners": [],
            "evidence_items": [
                {"id": "hublo_1", "kind": "page_signal:customer_archetype", "text": "Hospitals"},
                {"id": "hublo_2", "kind": "page_signal:workflow", "text": "Shift replacement"},
                {"id": "hublo_3", "kind": "page_signal:workflow", "text": "Internal mobility"},
                {"id": "hublo_4", "kind": "page_signal:capability", "text": "Pool management"},
                {"id": "hublo_5", "kind": "page_signal:service", "text": "API documentation"},
            ],
            "extracted_raw_phrases": [
                "Hospitals",
                "Care providers",
                "Healthcare staffing operations",
                "Shift replacement",
                "Internal mobility",
                "Pool management",
                "Replacement planning",
                "API documentation",
            ],
            "crawl_coverage": {"total_sites": 1, "total_pages": 2, "career_pages_selected": 1},
        },
        product_pages_found=2,
    )

    payload = build_company_context_artifacts(profile)
    taxonomy_by_layer: dict[str, list[str]] = {}
    for node in payload["taxonomy_nodes"]:
        taxonomy_by_layer.setdefault(node["layer"], []).append(node["phrase"])

    assert "Hospital" in taxonomy_by_layer["customer_archetype"] or "Hospitals" in taxonomy_by_layer["customer_archetype"]
    assert "Shift replacement" in taxonomy_by_layer["workflow"]
    assert "Internal mobility" in taxonomy_by_layer["workflow"]
    assert "Pool management" in taxonomy_by_layer["capability"]
    assert "API documentation" in taxonomy_by_layer["delivery_or_integration"]
    assert "Healthcare staffing operations" not in taxonomy_by_layer.get("capability", [])
    assert "AP-HP" not in taxonomy_by_layer.get("capability", [])
    assert any(
        lens["lens_type"] == "same_product_different_customer"
        for lens in (payload["sourcing_brief"].get("active_lenses") or [])
    )


def test_build_company_context_artifacts_marks_degraded_reasoning_when_llm_fails(monkeypatch):
    profile = CompanyProfile(
        workspace_id=12,
        buyer_company_url="https://4tpm.fr/platform/front-office",
                comparator_seed_urls=[],
        supporting_evidence_urls=[],
        comparator_seed_summaries={},
        geo_scope={},
        context_pack_json={
            "sites": [
                {
                    "url": "https://4tpm.fr/platform/front-office",
                    "company_name": "4TPM",
                    "summary": "",
                    "signals": [],
                    "customer_evidence": [],
                    "pages": [
                        {
                            "url": "https://4tpm.fr/platform/front-office",
                            "title": "4TPM - Plateforme Wealth Management",
                            "page_type": "product",
                            "blocks": [
                                {"type": "heading", "content": "Préparation et modélisation des portefeuilles", "level": 3},
                                {"type": "heading", "content": "Génération d'ordres blocs et routage full STP", "level": 3},
                            ],
                            "signals": [],
                            "customer_evidence": [],
                            "raw_content": "Front office titres PMS OMS REST API",
                        }
                    ],
                }
            ]
        },
        product_pages_found=1,
    )

    class _FailingOrchestrator:
        def run_stage(self, request):
            raise LLMOrchestrationError(
                "all routes failed",
                attempts=[
                    ModelAttemptTrace(
                        stage="market_map_reasoning",
                        provider="gemini",
                        model="gemini-2.0-flash",
                        latency_ms=10,
                        status="terminal_error",
                        retry_count=1,
                        error_message="schema invalid",
                    )
                ],
            )

    monkeypatch.setattr("app.services.company_context.LLMOrchestrator", _FailingOrchestrator)
    monkeypatch.setattr(
        "app.services.company_context.get_settings",
        lambda: type("S", (), {"gemini_api_key": "x", "openai_api_key": "", "anthropic_api_key": ""})(),
    )

    payload = build_company_context_artifacts(profile)

    assert payload["sourcing_brief"]["reasoning_status"] == "degraded"
    assert "deterministic fallback" in str(payload["sourcing_brief"]["reasoning_warning"] or "").lower()


def test_build_company_context_artifacts_keeps_reasoned_questions_capped(monkeypatch):
    profile = CompanyProfile(
        workspace_id=14,
        buyer_company_url="https://4tpm.fr/platform/front-office",
                comparator_seed_urls=[],
        supporting_evidence_urls=[],
        comparator_seed_summaries={},
        geo_scope={},
        context_pack_json={
            "sites": [
                {
                    "url": "https://4tpm.fr/platform/front-office",
                    "company_name": "4TPM",
                    "summary": "",
                    "signals": [],
                    "customer_evidence": [],
                    "pages": [
                        {
                            "url": "https://4tpm.fr/platform/front-office",
                            "title": "4TPM - Plateforme Wealth Management",
                            "page_type": "product",
                            "blocks": [
                                {"type": "heading", "content": "Front office titres pour gérants et desks de trading", "level": 1},
                                {"type": "heading", "content": "Préparation et modélisation des portefeuilles", "level": 3},
                            ],
                            "signals": [],
                            "customer_evidence": [],
                            "raw_content": "Front office titres PMS OMS",
                        }
                    ],
                }
            ]
        },
        product_pages_found=1,
    )

    class _FakeResponse:
        def __init__(self, text: str):
            self.text = text
            self.provider = "openai"
            self.model = "gpt-4.1-mini"

    class _FakeOrchestrator:
        def run_stage(self, request):
            import json

            prompt = request.prompt
            input_idx = prompt.index("Input:\n") + len("Input:\n")
            payload = json.loads(prompt[input_idx:])
            capability_id = next(
                node["id"] for node in payload["taxonomy_nodes"] if node["phrase"] == "Préparation et modélisation des portefeuilles"
            )
            return _FakeResponse(
                json.dumps(
                    {
                        "source_summary": "4TPM sells front-office wealth capabilities for trading desks and portfolio teams.",
                        "customer_node_ids": [],
                        "workflow_node_ids": [],
                        "capability_node_ids": [capability_id],
                        "delivery_or_integration_node_ids": [],
                        "active_lens_ids": [],
                        "adjacency_hypotheses": [],
                        "confidence_gaps": ["Need better named customer proof."],
                        "open_questions": [
                            "Which customer segment is strongest?",
                            "What adjacent workflow should be mapped next?",
                        ],
                    }
                )
            )

    monkeypatch.setattr("app.services.company_context.LLMOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(
        "app.services.company_context.get_settings",
        lambda: type("S", (), {"gemini_api_key": "x", "openai_api_key": "", "anthropic_api_key": ""})(),
    )

    payload = build_company_context_artifacts(profile)

    assert payload["sourcing_brief"]["reasoning_status"] == "success"
    assert "Which customer segment is strongest?" in payload["sourcing_brief"]["open_questions"]
    assert "What adjacent workflow should be mapped next?" in payload["sourcing_brief"]["open_questions"]
    assert len(payload["sourcing_brief"]["open_questions"]) <= 3


def test_build_company_context_artifacts_filters_strategy_style_reasoned_questions(monkeypatch):
    profile = CompanyProfile(
        workspace_id=15,
        buyer_company_url="https://4tpm.fr/platform/front-office",
                comparator_seed_urls=[],
        supporting_evidence_urls=[],
        comparator_seed_summaries={},
        geo_scope={},
        context_pack_json={
            "sites": [
                {
                    "url": "https://4tpm.fr/platform/front-office",
                    "company_name": "4TPM",
                    "summary": "",
                    "signals": [],
                    "customer_evidence": [],
                    "pages": [
                        {
                            "url": "https://4tpm.fr/platform/front-office",
                            "title": "4TPM - Plateforme Wealth Management",
                            "page_type": "product",
                            "blocks": [
                                {"type": "heading", "content": "Préparation et modélisation des portefeuilles", "level": 3},
                            ],
                            "signals": [],
                            "customer_evidence": [],
                            "raw_content": "Front office titres PMS OMS",
                        }
                    ],
                }
            ]
        },
        product_pages_found=1,
    )

    class _FakeResponse:
        def __init__(self, text: str):
            self.text = text
            self.provider = "openai"
            self.model = "gpt-4.1-mini"

    class _FakeOrchestrator:
        def run_stage(self, request):
            import json

            prompt = request.prompt
            input_idx = prompt.index("Input:\n") + len("Input:\n")
            payload = json.loads(prompt[input_idx:])
            capability_id = next(
                node["id"] for node in payload["taxonomy_nodes"] if node["phrase"] == "Préparation et modélisation des portefeuilles"
            )
            return _FakeResponse(
                json.dumps(
                    {
                        "source_summary": "4TPM sells front-office wealth capabilities.",
                        "customer_node_ids": [],
                        "workflow_node_ids": [],
                        "capability_node_ids": [capability_id],
                        "delivery_or_integration_node_ids": [],
                        "active_lens_ids": [],
                        "adjacency_hypotheses": [],
                        "confidence_gaps": [],
                        "open_questions": [
                            "Which customer segment is prioritized for growth?",
                            "What evidence clarifies workflow depth across front-office operations?",
                        ],
                    }
                )
            )

    monkeypatch.setattr("app.services.company_context.LLMOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(
        "app.services.company_context.get_settings",
        lambda: type("S", (), {"gemini_api_key": "x", "openai_api_key": "", "anthropic_api_key": ""})(),
    )

    payload = build_company_context_artifacts(profile)

    assert payload["sourcing_brief"]["reasoning_status"] == "success"
    assert "What evidence clarifies workflow depth across front-office operations?" in payload["sourcing_brief"]["open_questions"]
    assert "Which customer segment is prioritized for growth?" in payload["sourcing_brief"]["open_questions"]


def test_sourcing_brief_reasoning_prompt_stays_domain_agnostic():
    prompt = _sourcing_brief_reasoning_prompt(
        {
            "prompt_version": "v2",
            "source_company": {"name": "ExampleCo", "website": "https://example.com"},
            "crawl_coverage": {"total_pages": 4},
            "taxonomy_nodes": [],
            "ranked_nodes_by_layer": {
                "customer_archetype": [],
                "workflow": [],
                "capability": [],
                "delivery_or_integration": [],
            },
            "lens_seeds": [],
            "named_customer_proof": [],
            "integration_partner_proof": [],
            "evidence_highlights": {
                "named_customer_names": [],
                "integration_partner_names": [],
                "top_capability_phrases": [],
                "top_workflow_phrases": [],
                "top_customer_phrases": [],
                "top_delivery_phrases": [],
            },
            "selection_rules": {"summary_max_words": 120},
            "fallback_brief": {
                "source_summary": "",
                "customer_node_ids": [],
                "workflow_node_ids": [],
                "capability_node_ids": [],
                "delivery_or_integration_node_ids": [],
                "active_lens_ids": [],
                "confidence_gaps": [],
                "open_questions": [],
            },
        }
    )

    lowered = prompt.lower()
    assert "use the source company's own vocabulary" in lowered
    assert "opening paragraph of the sourcing brief" in lowered
    assert "what adjacency box it suggests" in lowered
    assert "4tpm" not in lowered
    assert "hublo" not in lowered
    assert "private bank" not in lowered
    assert "hospital" not in lowered
