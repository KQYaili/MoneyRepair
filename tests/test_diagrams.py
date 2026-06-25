import json

from moneyrepair.diagrams import production_pipeline_spec, render_diagram_svg, write_diagram


def test_production_pipeline_spec_has_loop():
    spec = production_pipeline_spec()
    assert len(spec.nodes) == 7
    node_ids = {node.id for node in spec.nodes}
    assert {"acq", "manifest", "prune", "search", "report", "review", "confirm"} == node_ids

    feedback = [edge for edge in spec.edges if edge.kind == "feedback"]
    assert len(feedback) == 1
    assert feedback[0].source == "review"
    assert feedback[0].target == "search"


def test_render_diagram_svg_keeps_editable_text():
    spec = production_pipeline_spec()
    svg = render_diagram_svg(spec)
    assert svg.startswith("<svg")
    assert "<text" in svg
    assert "Operator review" in svg
    assert "marker-end" in svg


def test_write_diagram_writes_spec_and_svg(tmp_path):
    spec = production_pipeline_spec()
    outputs = write_diagram(spec, tmp_path / "pipeline")

    spec_path = tmp_path / "pipeline.json"
    svg_path = tmp_path / "pipeline.svg"
    assert spec_path.exists()
    assert svg_path.exists()
    assert outputs["spec"] == str(spec_path)

    reloaded = json.loads(spec_path.read_text(encoding="utf-8"))
    assert len(reloaded["nodes"]) == 7
    assert reloaded["title"] == spec.title
