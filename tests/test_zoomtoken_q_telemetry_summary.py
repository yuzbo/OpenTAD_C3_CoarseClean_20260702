from tools.bata.summarize_zoomtoken_q_telemetry import summarize


def test_summary_reports_dynamic_budget_and_geometry():
    route = {
        "k_t": {"values": [0, 2, 4, 2]},
        "geometry": {
            "area": {"mean": 0.25},
            "width_floor_saturation_rate": 0.5,
            "height_floor_saturation_rate": 0.25,
        },
        "roles": {
            "aggregate_counts": {"context": 2, "roi": 4, "residual": 2}
        },
    }
    summary = summarize(
        {"dataset_count": 2, "records": [{"route": route}, {"route": route}]}
    )
    assert summary["record_count"] == 2
    assert summary["k_t"]["min"] == 0
    assert summary["k_t"]["max"] == 4
    assert summary["k_t"]["zero_fraction"] == 0.25
    assert summary["geometry"]["area_mean"] == 0.25
    assert summary["role_fractions"] == {
        "context": 0.25,
        "roi": 0.5,
        "residual": 0.25,
    }

