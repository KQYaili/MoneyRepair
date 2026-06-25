from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from pathlib import Path

# Visio-style editable diagrams. Following the editable-spec approach used by the
# referenced Visio skills, the spec is an explicit node/edge graph with geometry
# and style so it can be imported into Visio/drawio, and the SVG keeps real
# <text> elements so labels stay editable in any vector editor.

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
    kind: str = "flow"  # flow | feedback

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
    """Editable schematic of the v2.0 production reconstruction loop."""

    step_w, step_h, gap, top = 150.0, 56.0, 40.0, 70.0
    steps = [
        ("acq", "Acquisition QA", "terminator"),
        ("manifest", "Manifest model", "data"),
        ("prune", "Compatibility pruning", "process"),
        ("search", "DFS search", "process"),
        ("report", "Candidate report", "data"),
        ("review", "Operator review", "decision"),
        ("confirm", "Confirmed note", "terminator"),
    ]
    nodes: list[DiagramNode] = []
    x = 30.0
    for node_id, label, style in steps:
        nodes.append(DiagramNode(id=node_id, label=label, x=x, y=top, width=step_w, height=step_h, style=style))
        x += step_w + gap

    edges = [
        DiagramEdge("acq", "manifest", "accepted frames"),
        DiagramEdge("manifest", "prune"),
        DiagramEdge("prune", "search", "compatible pairs"),
        DiagramEdge("search", "report", "candidates"),
        DiagramEdge("report", "review"),
        DiagramEdge("review", "confirm", "accept"),
        DiagramEdge("review", "search", "reject / retry", kind="feedback"),
    ]
    width = x - gap + 30.0
    height = top + step_h + 70.0
    return DiagramSpec(title="MoneyRepair production pipeline", nodes=nodes, edges=edges, width=width, height=height)


DIAGRAMS = {"production-pipeline": production_pipeline_spec}


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
    if edge.kind == "feedback":
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

    x0 = source.x + source.width
    y0 = source.y + source.height / 2
    x1 = target.x
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


def write_diagram(spec: DiagramSpec, output_prefix: str | Path) -> dict[str, str]:
    """Write the editable JSON spec and an editable-text SVG for a diagram."""

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    spec_path = output_prefix.with_suffix(".json")
    svg_path = output_prefix.with_suffix(".svg")
    spec_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
    svg_path.write_text(render_diagram_svg(spec), encoding="utf-8")
    return {"spec": str(spec_path), "svg": str(svg_path)}
