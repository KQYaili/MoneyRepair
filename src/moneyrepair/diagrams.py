from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html import escape
from pathlib import Path

# The JSON graph is the deterministic source, Draw.io is the canonical editable
# artifact, and SVG is the review-friendly rendered form. VSDX remains an
# explicit compatibility export rather than a default dependency.

_STYLE_FILL = {
    "terminator": "#4C78A8",
    "process": "#72B7B2",
    "data": "#F2B441",
    "decision": "#B279A2",
}
_TEXT_COLOR = "#10222e"


@dataclass(frozen=True)
class DiagramNode:
    id: str
    label: str
    x: float
    y: float
    width: float = 150.0
    height: float = 56.0
    style: str = "process"  # terminator | process | data | decision

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "style": self.style,
        }


@dataclass(frozen=True)
class DiagramEdge:
    source: str
    target: str
    label: str = ""
    kind: str = "flow"  # flow | feedback | reject

    def to_dict(self) -> dict:
        return {"source": self.source, "target": self.target, "label": self.label, "kind": self.kind}


@dataclass(frozen=True)
class DiagramSpec:
    title: str
    nodes: list[DiagramNode]
    edges: list[DiagramEdge]
    width: float
    height: float

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "width": self.width,
            "height": self.height,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }


def production_pipeline_spec() -> DiagramSpec:
    """Current reconstruction path, including the physical handoff boundary."""

    step_w, step_h, top, bottom = 158.0, 60.0, 72.0, 188.0
    nodes = [
        DiagramNode("contract", "Frozen capture contract", 30.0, top, step_w, step_h, "data"),
        DiagramNode("ingest", "Mask + image ingest", 232.0, top, step_w, step_h, "process"),
        DiagramNode("pose", "Pose + uncertainty", 434.0, top, step_w, step_h, "process"),
        DiagramNode("tear", "Placed tear evidence", 636.0, top, step_w, step_h, "process"),
        DiagramNode("candidates", "Core + gap candidates", 636.0, bottom, step_w, step_h, "process"),
        DiagramNode("cover", "Exact-cover selection", 434.0, bottom, step_w, step_h, "process"),
        DiagramNode("route", "Automatic or review?", 232.0, bottom, step_w, step_h, "decision"),
        DiagramNode("confirm", "Confirmed assembly", 30.0, bottom, step_w, step_h, "terminator"),
    ]

    edges = [
        DiagramEdge("contract", "ingest"),
        DiagramEdge("ingest", "pose"),
        DiagramEdge("pose", "tear"),
        DiagramEdge("tear", "candidates"),
        DiagramEdge("candidates", "cover"),
        DiagramEdge("cover", "route"),
        DiagramEdge("route", "confirm"),
        DiagramEdge("route", "pose", "review / correct", kind="feedback"),
    ]
    width = 824.0
    height = 350.0
    return DiagramSpec(title="MoneyRepair production pipeline", nodes=nodes, edges=edges, width=width, height=height)


def acquisition_flow_spec() -> DiagramSpec:
    """Physical acquisition contract and truth-isolated pose handoff."""
    step_w, step_h, top, bottom = 158.0, 60.0, 72.0, 230.0
    nodes = [
        DiagramNode("contract", "Coordinate contract", 30.0, top, step_w, step_h, "data"),
        DiagramNode("capture", "Scanner / phone capture", 232.0, top, step_w, step_h, "process"),
        DiagramNode("segment", "Foreground mask", 434.0, top, step_w, step_h, "process"),
        DiagramNode("mask_gate", "Mask tolerance passes?", 636.0, top, step_w, step_h, "decision"),
        DiagramNode("locate", "Top-k pose + uncertainty", 636.0, bottom, step_w, step_h, "process"),
        DiagramNode("pose_gate", "Pose gate passes?", 434.0, bottom, step_w, step_h, "decision"),
        DiagramNode("handoff", "Placed-fragment handoff", 232.0, bottom, step_w, step_h, "terminator"),
    ]

    edges = [
        DiagramEdge("contract", "capture"),
        DiagramEdge("capture", "segment"),
        DiagramEdge("segment", "mask_gate"),
        DiagramEdge("mask_gate", "locate"),
        DiagramEdge("locate", "pose_gate"),
        DiagramEdge("pose_gate", "handoff"),
        DiagramEdge("mask_gate", "capture", "recapture", kind="feedback"),
        DiagramEdge("pose_gate", "locate", "review", kind="feedback"),
    ]
    width = 824.0
    height = 390.0
    return DiagramSpec(title="MoneyRepair acquisition flow", nodes=nodes, edges=edges, width=width, height=height)


def search_logic_spec() -> DiagramSpec:
    """Frozen deterministic candidate and exact-cover path."""
    step_w, step_h, top = 150.0, 56.0, 70.0
    nodes = [
        DiagramNode("start", "High-confidence core seeds", 30.0, top, step_w, step_h, "terminator"),
        DiagramNode("expand", "Bounded core expansion", 220.0, top, step_w, step_h, "process"),
        DiagramNode("gap", "Residual-gap proposals", 410.0, top, step_w, step_h, "process"),
        DiagramNode("cover", "Exact-cover selection", 600.0, top, step_w, step_h, "process"),
        DiagramNode("gate", "Evidence gate passes?", 790.0, top, step_w, step_h, "decision"),
        DiagramNode("auto", "Automatic confirmation", 980.0, top, step_w, step_h, "terminator"),
        DiagramNode("review", "Human review queue", 790.0, top + step_h + 50.0, step_w, step_h, "data"),
    ]
    edges = [
        DiagramEdge("start", "expand"),
        DiagramEdge("expand", "gap"),
        DiagramEdge("gap", "cover"),
        DiagramEdge("cover", "gate"),
        DiagramEdge("gate", "auto", "yes"),
        DiagramEdge("gate", "review", "no / ambiguous"),
    ]
    width = 980.0 + step_w + 30.0
    height = top + 2 * step_h + 50.0 + 70.0
    return DiagramSpec(title="MoneyRepair search logic", nodes=nodes, edges=edges, width=width, height=height)


def operator_loop_spec() -> DiagramSpec:
    """Editable schematic of the operator review loop."""
    step_w, step_h, top = 150.0, 56.0, 70.0
    nodes = [
        DiagramNode("start", "Start", 30.0, top, step_w, step_h, "terminator"),
        DiagramNode("gallery", "Render gallery", 220.0, top, step_w, step_h, "data"),
        DiagramNode("review", "Operator review", 410.0, top, step_w, step_h, "decision"),
        DiagramNode("confirm", "Confirm reconstruction", 600.0, top, step_w, step_h, "process"),
        DiagramNode("remove", "Remove from pool", 790.0, top, step_w, step_h, "process"),
        DiagramNode("next", "Next batch", 980.0, top, step_w, step_h, "terminator"),
        DiagramNode("reject", "Reject candidate", 410.0, top + step_h + 50.0, step_w, step_h, "process"),
    ]
    edges = [
        DiagramEdge("start", "gallery"),
        DiagramEdge("gallery", "review"),
        DiagramEdge("review", "confirm", "accept"),
        DiagramEdge("review", "reject", "reject"),
        DiagramEdge("confirm", "remove"),
        DiagramEdge("remove", "next"),
        DiagramEdge("reject", "gallery", "retry", kind="feedback"),
    ]
    width = 980.0 + step_w + 30.0
    height = top + 2 * step_h + 50.0 + 70.0
    return DiagramSpec(title="MoneyRepair operator loop", nodes=nodes, edges=edges, width=width, height=height)


def research_gates_spec() -> DiagramSpec:
    """Evidence gates that prevent simulation tuning from becoming production claims."""

    step_w, step_h, top = 166.0, 62.0, 70.0
    nodes = [
        DiagramNode("capture", "Physical capture set", 30.0, top, step_w, step_h, "data"),
        DiagramNode("mask", "Gate 0: mask contract", 236.0, top, step_w, step_h, "decision"),
        DiagramNode("pose", "Gate 1: pose handoff", 442.0, top, step_w, step_h, "decision"),
        DiagramNode("failure", "Gate 2: failure localization", 648.0, top, step_w, step_h, "decision"),
        DiagramNode("component", "Deterministic component A/B", 854.0, top, step_w, step_h, "process"),
        DiagramNode("learned", "Conditional seam descriptor", 1060.0, top, step_w, step_h, "process"),
        DiagramNode("freeze", "Keep core frozen", 648.0, top + step_h + 58.0, step_w, step_h, "terminator"),
    ]
    edges = [
        DiagramEdge("capture", "mask"),
        DiagramEdge("mask", "pose"),
        DiagramEdge("pose", "failure"),
        DiagramEdge("failure", "component"),
        DiagramEdge("component", "learned"),
        DiagramEdge("mask", "freeze", kind="reject"),
        DiagramEdge("pose", "freeze", "gate fails", kind="reject"),
        DiagramEdge("failure", "freeze", "no qualifying wall"),
    ]
    width = 1060.0 + step_w + 30.0
    height = top + 2 * step_h + 58.0 + 70.0
    return DiagramSpec(
        title="MoneyRepair evidence-gated research path", nodes=nodes, edges=edges, width=width, height=height
    )


DIAGRAMS = {
    "production-pipeline": production_pipeline_spec,
    "acquisition-flow": acquisition_flow_spec,
    "search-logic": search_logic_spec,
    "operator-loop": operator_loop_spec,
    "research-gates": research_gates_spec,
}


def _wrap(label: str, limit: int = 15) -> list[str]:
    words = label.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > limit and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [label]


def _node_shape_svg(node: DiagramNode) -> str:
    x, y, w, h = node.x, node.y, node.width, node.height
    fill = _STYLE_FILL.get(node.style, "#72B7B2")
    if node.style == "terminator":
        shape = f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{h / 2:.1f}" ry="{h / 2:.1f}" fill="{fill}" stroke="#10222e" stroke-width="1.4"/>'
    elif node.style == "data":
        skew = 16.0
        points = f"{x + skew:.1f},{y:.1f} {x + w:.1f},{y:.1f} {x + w - skew:.1f},{y + h:.1f} {x:.1f},{y + h:.1f}"
        shape = f'<polygon points="{points}" fill="{fill}" stroke="#10222e" stroke-width="1.4"/>'
    elif node.style == "decision":
        cx, cy = x + w / 2, y + h / 2
        points = f"{cx:.1f},{y:.1f} {x + w:.1f},{cy:.1f} {cx:.1f},{y + h:.1f} {x:.1f},{cy:.1f}"
        shape = f'<polygon points="{points}" fill="{fill}" stroke="#10222e" stroke-width="1.4"/>'
    else:
        shape = f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" ry="8" fill="{fill}" stroke="#10222e" stroke-width="1.4"/>'

    lines = _wrap(node.label)
    cx, cy = x + w / 2, y + h / 2
    start_y = cy - (len(lines) - 1) * 6.5
    spans = "".join(
        f'<tspan x="{cx:.1f}" y="{start_y + i * 13:.1f}">{escape(line)}</tspan>' for i, line in enumerate(lines)
    )
    text = f'<text text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="11" font-weight="600" fill="{_TEXT_COLOR}">{spans}</text>'
    return shape + text


def _edge_svg(spec_nodes: dict[str, DiagramNode], edge: DiagramEdge) -> str:
    source = spec_nodes[edge.source]
    target = spec_nodes[edge.target]
    if edge.kind == "reject":
        x0 = source.x + source.width / 2
        y0 = source.y + source.height
        x1 = target.x + target.width / 2
        y1 = target.y
        line = f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#E45756" stroke-width="1.6" stroke-dasharray="6 4" marker-end="url(#arrow-fb)"/>'
        label = ""
        if edge.label:
            label = f'<text x="{(x0 + x1) / 2:.1f}" y="{(y0 + y1) / 2 - 5:.1f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="9" fill="#E45756">{escape(edge.label)}</text>'
        return line + label
    if edge.kind == "feedback":
        if source.y >= target.y:
            # Source is below target, route it downwards/below
            x0 = source.x + source.width / 2
            y0 = source.y + source.height
            x1 = target.x + target.width / 2
            y1 = target.y + target.height
            peak = max(y0, y1) + 40.0
            path = f"M {x0:.1f} {y0:.1f} C {x0:.1f} {peak:.1f}, {x1:.1f} {peak:.1f}, {x1:.1f} {y1:.1f}"
            line = f'<path d="{path}" fill="none" stroke="#E45756" stroke-width="1.6" stroke-dasharray="6 4" marker-end="url(#arrow-fb)"/>'
            label = ""
            if edge.label:
                label = f'<text x="{(x0 + x1) / 2:.1f}" y="{peak + 12:.1f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="9" fill="#E45756">{escape(edge.label)}</text>'
            return line + label
        else:
            x0 = source.x + source.width / 2
            y0 = source.y
            x1 = target.x + target.width / 2
            y1 = target.y
            peak = min(y0, y1) - 44.0
            path = f"M {x0:.1f} {y0:.1f} C {x0:.1f} {peak:.1f}, {x1:.1f} {peak:.1f}, {x1:.1f} {y1:.1f}"
            line = f'<path d="{path}" fill="none" stroke="#E45756" stroke-width="1.6" stroke-dasharray="6 4" marker-end="url(#arrow-fb)"/>'
            label = ""
            if edge.label:
                label = f'<text x="{(x0 + x1) / 2:.1f}" y="{peak - 4:.1f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="9" fill="#E45756">{escape(edge.label)}</text>'
            return line + label

    # Standard flow edge
    # Check if target is directly below source (aligned vertically)
    if abs(source.x - target.x) < 40.0 and target.y > source.y:
        x0 = source.x + source.width / 2
        y0 = source.y + source.height
        x1 = target.x + target.width / 2
        y1 = target.y
        line = f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#10222e" stroke-width="1.6" marker-end="url(#arrow)"/>'
        label = ""
        if edge.label:
            label = f'<text x="{x0 + 8:.1f}" y="{(y0 + y1) / 2:.1f}" text-anchor="start" font-family="Arial, Helvetica, sans-serif" font-size="9" fill="#41525c">{escape(edge.label)}</text>'
        return line + label

    if target.x >= source.x:
        x0 = source.x + source.width
        x1 = target.x
    else:
        x0 = source.x
        x1 = target.x + target.width
    y0 = source.y + source.height / 2
    y1 = target.y + target.height / 2
    line = f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#10222e" stroke-width="1.6" marker-end="url(#arrow)"/>'
    label = ""
    if edge.label:
        label = f'<text x="{(x0 + x1) / 2:.1f}" y="{y0 - 5:.1f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="9" fill="#41525c">{escape(edge.label)}</text>'
    return line + label


def render_diagram_svg(spec: DiagramSpec) -> str:
    nodes = {node.id: node for node in spec.nodes}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {spec.width:.0f} {spec.height:.0f}" width="{spec.width:.0f}" height="{spec.height:.0f}">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#10222e"/></marker>',
        '<marker id="arrow-fb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#E45756"/></marker>',
        "</defs>",
        f'<rect x="0" y="0" width="{spec.width:.0f}" height="{spec.height:.0f}" fill="#ffffff"/>',
        f'<text x="{spec.width / 2:.1f}" y="34" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="16" font-weight="700" fill="{_TEXT_COLOR}">{escape(spec.title)}</text>',
    ]
    for edge in spec.edges:
        parts.append(_edge_svg(nodes, edge))
    for node in spec.nodes:
        parts.append(_node_shape_svg(node))
    parts.append("</svg>")
    return "\n".join(parts)


def _drawio_node_style(node: DiagramNode) -> str:
    base = (
        "whiteSpace=wrap;html=1;strokeColor=#10222e;strokeWidth=1.4;"
        f"fillColor={_STYLE_FILL.get(node.style, '#72B7B2')};"
        f"fontColor={_TEXT_COLOR};fontStyle=1;fontSize=11;"
    )
    if node.style == "terminator":
        return base + "rounded=1;arcSize=50;"
    if node.style == "data":
        return base + "shape=parallelogram;perimeter=parallelogramPerimeter;"
    if node.style == "decision":
        return base + "rhombus;"
    return base + "rounded=1;arcSize=8;"


def render_diagram_drawio(spec: DiagramSpec) -> str:
    """Render an uncompressed diagrams.net document with editable cells."""

    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "agent": "MoneyRepair",
            "version": "24.7.17",
            "compressed": "false",
        },
    )
    diagram = ET.SubElement(mxfile, "diagram", {"id": "moneyrepair", "name": "Page-1"})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": str(int(spec.width)),
            "dy": str(int(spec.height)),
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": str(int(spec.width)),
            "pageHeight": str(int(spec.height)),
            "math": "0",
            "shadow": "0",
        },
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
    title = ET.SubElement(
        root,
        "mxCell",
        {
            "id": "title",
            "value": spec.title,
            "style": (
                "text;html=1;align=center;verticalAlign=middle;resizable=0;"
                "points=[];autosize=1;strokeColor=none;fillColor=none;"
                f"fontColor={_TEXT_COLOR};fontSize=16;fontStyle=1;"
            ),
            "vertex": "1",
            "parent": "1",
        },
    )
    ET.SubElement(
        title,
        "mxGeometry",
        {"x": "20", "y": "12", "width": str(spec.width - 40), "height": "30", "as": "geometry"},
    )
    for node in spec.nodes:
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": f"node-{node.id}",
                "value": node.label,
                "style": _drawio_node_style(node),
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": f"{node.x:.1f}",
                "y": f"{node.y:.1f}",
                "width": f"{node.width:.1f}",
                "height": f"{node.height:.1f}",
                "as": "geometry",
            },
        )
    for index, edge in enumerate(spec.edges):
        color = "#E45756" if edge.kind in {"feedback", "reject"} else "#10222e"
        dashed = "1" if edge.kind in {"feedback", "reject"} else "0"
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": f"edge-{index}",
                "value": edge.label,
                "style": (
                    "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
                    "jettySize=auto;html=1;endArrow=block;endFill=1;"
                    f"strokeColor={color};strokeWidth=1.6;dashed={dashed};"
                    "fontSize=9;labelBackgroundColor=#ffffff;"
                ),
                "edge": "1",
                "parent": "1",
                "source": f"node-{edge.source}",
                "target": f"node-{edge.target}",
            },
        )
        ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    ET.indent(mxfile, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(mxfile, encoding="unicode")


def write_diagram(
    spec: DiagramSpec,
    output_prefix: str | Path,
    *,
    export_vsdx_file: bool = False,
) -> dict[str, str]:
    """Write JSON, editable Draw.io, and SVG artifacts.

    VSDX is an opt-in compatibility export because it requires local Visio COM.
    """

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    spec_path = output_prefix.with_suffix(".json")
    drawio_path = output_prefix.with_suffix(".drawio")
    svg_path = output_prefix.with_suffix(".svg")
    spec_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
    drawio_path.write_text(render_diagram_drawio(spec), encoding="utf-8")
    svg_path.write_text(render_diagram_svg(spec), encoding="utf-8")

    outputs = {
        "spec": str(spec_path),
        "drawio": str(drawio_path),
        "svg": str(svg_path),
    }

    if export_vsdx_file:
        vsdx_path = output_prefix.with_suffix(".vsdx")
        export_to_vsdx(spec, vsdx_path)
        outputs["vsdx"] = str(vsdx_path)

    return outputs


def hex_to_rgb_formula(hex_str: str) -> str:
    hex_str = hex_str.lstrip("#")
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return f"RGB({r}, {g}, {b})"


def export_to_vsdx(spec: DiagramSpec, output_path: Path | str) -> None:
    """Automate Microsoft Visio via COM to draw the diagram and save as VSDX."""
    import win32com.client
    import array
    import os

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        visio = win32com.client.Dispatch("Visio.Application")
    except Exception as e:
        raise RuntimeError(f"Failed to start Microsoft Visio: {e}")

    visio.Visible = False

    try:
        # Create a new document without a template (blank page)
        doc = visio.Documents.Add("")
        page = doc.Pages.Item(1)

        scale = 0.015  # inches per pixel
        page_w = spec.width * scale
        page_h = spec.height * scale

        # Set page dimensions
        page.PageSheet.Cells("PageWidth").FormulaU = f"{page_w:.4f} in"
        page.PageSheet.Cells("PageHeight").FormulaU = f"{page_h:.4f} in"

        # 1. Add title
        title_y = page_h - 0.5
        title_shape = page.DrawRectangle(0.5, title_y - 0.4, page_w - 0.5, title_y + 0.1)
        title_shape.Text = spec.title
        title_shape.Cells("LinePattern").FormulaU = "0"  # No line
        title_shape.Cells("FillPattern").FormulaU = "0"  # No fill
        try:
            title_shape.Cells("Char.Size").FormulaU = "16 pt"
            title_shape.Cells("Char.Style").FormulaU = "1"  # Bold
        except Exception:
            pass

        # 2. Draw nodes
        drawn_shapes = {}
        for node in spec.nodes:
            x1 = node.x * scale
            x2 = (node.x + node.width) * scale
            y1 = (spec.height - (node.y + node.height)) * scale
            y2 = (spec.height - node.y) * scale

            cx = (node.x + node.width / 2.0) * scale
            cy = (spec.height - (node.y + node.height / 2.0)) * scale
            w_half = (node.width / 2.0) * scale
            h_half = (node.height / 2.0) * scale

            fill_hex = _STYLE_FILL.get(node.style, "#72B7B2")
            fill_formula = hex_to_rgb_formula(fill_hex)

            if node.style == "decision":
                # Draw diamond
                points = array.array('d', [
                    cx, cy + h_half,
                    cx + w_half, cy,
                    cx, cy - h_half,
                    cx - w_half, cy,
                    cx, cy + h_half
                ])
                shape = page.DrawPolyline(points, 0)
            elif node.style == "data":
                # Draw skewed parallelogram
                skew = 16.0 * scale
                points = array.array('d', [
                    x1 + skew, y2,
                    x2, y2,
                    x2 - skew, y1,
                    x1, y1,
                    x1 + skew, y2
                ])
                shape = page.DrawPolyline(points, 0)
            elif node.style == "terminator":
                # Rounded rect (pill)
                shape = page.DrawRectangle(x1, y1, x2, y2)
                rounding = (node.height / 2.0) * scale
                try:
                    shape.Cells("Rounding").FormulaU = f"{rounding:.4f} in"
                except Exception:
                    pass
            else:  # process / standard
                shape = page.DrawRectangle(x1, y1, x2, y2)
                try:
                    shape.Cells("Rounding").FormulaU = "0.08 in"
                except Exception:
                    pass

            shape.Text = node.label
            shape.Cells("FillForegnd").FormulaU = fill_formula
            shape.Cells("LineColor").FormulaU = "RGB(16, 34, 46)"
            shape.Cells("LineWeight").FormulaU = "1.5 pt"
            try:
                shape.Cells("Char.Size").FormulaU = "10 pt"
                shape.Cells("Char.Font").FormulaU = "Segoe UI"
            except Exception:
                pass

            drawn_shapes[node.id] = shape

        # 3. Connect nodes
        connector_tool = visio.ConnectorToolDataObject
        for edge in spec.edges:
            source_shp = drawn_shapes.get(edge.source)
            target_shp = drawn_shapes.get(edge.target)
            if not source_shp or not target_shp:
                continue

            conn = page.Drop(connector_tool, 0.0, 0.0)
            if edge.label:
                conn.Text = edge.label

            # Connect using GlueTo
            conn.Cells("BeginX").GlueTo(source_shp.Cells("PinX"))
            conn.Cells("EndX").GlueTo(target_shp.Cells("PinX"))

            # Format connector
            if edge.kind == "feedback":
                conn.Cells("LineColor").FormulaU = "RGB(228, 87, 86)"
                try:
                    conn.Cells("LinePattern").FormulaU = "2"  # Dashed
                except Exception:
                    pass
            else:
                conn.Cells("LineColor").FormulaU = "RGB(16, 34, 46)"

            conn.Cells("LineWeight").FormulaU = "1.5 pt"

        # Save document
        if output_path.exists():
            os.remove(output_path)
        doc.SaveAs(str(output_path.absolute()))
        doc.Close()

    finally:
        visio.Quit()
