from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
from PIL import Image

from moneyrepair.solver import CoverageSolution
from moneyrepair.types import Fragment


PALETTE = np.array(
    [
        [230, 68, 68],
        [46, 154, 112],
        [55, 116, 209],
        [242, 180, 65],
        [150, 86, 188],
        [42, 180, 205],
        [236, 112, 52],
        [99, 150, 52],
    ],
    dtype=np.uint8,
)


def render_solution(
    template: np.ndarray,
    fragments: list[Fragment],
    solution: CoverageSolution,
    path: str | Path,
    alpha: float = 0.42,
) -> None:
    lookup = {fragment.id: fragment for fragment in fragments}
    canvas = template.astype(np.float32).copy()
    for index, fragment_id in enumerate(solution.fragment_ids):
        fragment = lookup[fragment_id]
        color = PALETTE[index % len(PALETTE)].astype(np.float32)
        mask = fragment.mask
        canvas[mask] = canvas[mask] * (1.0 - alpha) + color * alpha
    output = np.clip(canvas, 0, 255).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="RGB").save(path)


def render_solution_gallery(
    template: np.ndarray,
    fragments: list[Fragment],
    solutions: list[CoverageSolution],
    output_dir: str | Path,
    limit: int = 20,
) -> list[Path]:
    output_dir = Path(output_dir)
    image_paths: list[Path] = []
    for index, solution in enumerate(solutions[:limit]):
        path = output_dir / f"solution_{index:03d}.png"
        render_solution(template, fragments, solution, path)
        image_paths.append(path)
    return image_paths


def write_solution_report(
    solutions: list[CoverageSolution],
    image_paths: list[Path],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for index, (solution, image_path) in enumerate(zip(solutions, image_paths)):
        try:
            image_ref = image_path.relative_to(path.parent).as_posix()
        except ValueError:
            image_ref = image_path.as_posix()
        ids = ", ".join(escape(fragment_id) for fragment_id in solution.fragment_ids)
        rows.append(
            "\n".join(
                [
                    '<article class="candidate">',
                    f'<a href="{escape(image_ref)}"><img src="{escape(image_ref)}" alt="solution {index}"></a>',
                    f"<h2>Candidate {index + 1}</h2>",
                    f"<p>Coverage: {solution.coverage:.4%}</p>",
                    f"<p>Fragments: {len(solution.fragment_ids)}</p>",
                    f"<details><summary>Fragment ids</summary><p>{ids}</p></details>",
                    "</article>",
                ]
            )
        )
    html = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>MoneyRepair Candidates</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f4;color:#222}",
            "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}",
            ".candidate{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px}",
            "img{width:100%;height:auto;border:1px solid #ddd;background:#eee}",
            "h1{font-size:24px;margin:0 0 18px}",
            "h2{font-size:18px;margin:10px 0 6px}",
            "p{margin:5px 0}",
            "details p{word-break:break-word;font-size:13px}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>MoneyRepair Candidates</h1>",
            "<main>",
            *rows,
            "</main>",
            "</body>",
            "</html>",
        ]
    )
    path.write_text(html, encoding="utf-8")
