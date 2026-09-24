import json

from moneyrepair.diagrams import (
    production_pipeline_spec,
    acquisition_flow_spec,
    search_logic_spec,
    operator_loop_spec,
    research_gates_spec,
    render_diagram_drawio,
    render_diagram_svg,
    write_diagram,
)


def test_production_pipeline_spec_has_loop():
    spec = production_pipeline_spec()
    assert len(spec.nodes) == 8
    node_ids = {node.id for node in spec.nodes}
    assert {
        "contract",
        "ingest",
        "pose",
        "tear",
        "candidates",
        "cover",
        "route",
        "confirm",
    } == node_ids

    feedback = [edge for edge in spec.edges if edge.kind == "feedback"]
    assert len(feedback) == 1
    assert feedback[0].source == "route"
    assert feedback[0].target == "pose"


def test_acquisition_flow_spec():
    spec = acquisition_flow_spec()
    assert len(spec.nodes) == 7
    node_ids = {node.id for node in spec.nodes}
    assert {"contract", "capture", "segment", "mask_gate", "locate", "pose_gate", "handoff"} == node_ids


def test_search_logic_spec():
    spec = search_logic_spec()
    assert len(spec.nodes) == 7
    node_ids = {node.id for node in spec.nodes}
    assert {"start", "expand", "gap", "cover", "gate", "auto", "review"} == node_ids


def test_operator_loop_spec():
    spec = operator_loop_spec()
    assert len(spec.nodes) == 7
    node_ids = {node.id for node in spec.nodes}
    assert {"start", "gallery", "review", "confirm", "remove", "next", "reject"} == node_ids
    feedback = [edge for edge in spec.edges if edge.kind == "feedback"]
    assert len(feedback) == 1
    assert feedback[0].source == "reject"
    assert feedback[0].target == "gallery"


def test_research_gates_spec_keeps_algorithm_work_conditional():
    spec = research_gates_spec()
    node_ids = {node.id for node in spec.nodes}
    assert {"capture", "mask", "pose", "failure", "component", "learned", "freeze"} == node_ids
    assert any(edge.source == "failure" and edge.target == "component" for edge in spec.edges)
    assert sum(edge.kind == "reject" for edge in spec.edges) == 2


def test_render_diagram_svg_keeps_editable_text():
    spec = production_pipeline_spec()
    svg = render_diagram_svg(spec)
    assert svg.startswith("<svg")
    assert "<text" in svg
    assert "Confirmed" in svg
    assert "marker-end" in svg


def test_render_diagram_drawio_keeps_editable_cells():
    spec = production_pipeline_spec()
    drawio = render_diagram_drawio(spec)
    assert drawio.startswith('<?xml version="1.0"')
    assert "<mxGraphModel" in drawio
    assert 'id="node-contract"' in drawio
    assert "Frozen capture contract" in drawio


def test_write_diagram_writes_spec_drawio_and_svg(tmp_path):
    spec = production_pipeline_spec()
    outputs = write_diagram(spec, tmp_path / "pipeline")

    spec_path = tmp_path / "pipeline.json"
    drawio_path = tmp_path / "pipeline.drawio"
    svg_path = tmp_path / "pipeline.svg"
    assert spec_path.exists()
    assert drawio_path.exists()
    assert svg_path.exists()
    assert outputs["spec"] == str(spec_path)
    assert outputs["drawio"] == str(drawio_path)
    assert "vsdx" not in outputs

    reloaded = json.loads(spec_path.read_text(encoding="utf-8"))
    assert len(reloaded["nodes"]) == 8
    assert reloaded["title"] == spec.title
