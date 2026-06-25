import numpy as np

from moneyrepair.solver import CoverageSolution
from moneyrepair.types import Fragment
from moneyrepair.visualize import render_solution_gallery, write_solution_report


def test_write_solution_report_links_rendered_images(tmp_path):
    template = np.zeros((6, 8, 3), dtype=np.uint8)
    mask = np.zeros((6, 8), dtype=bool)
    mask[1:5, 2:6] = True
    fragment = Fragment("frag-001", mask, image=np.where(mask[..., None], 255, 0).astype(np.uint8))
    solution = CoverageSolution(fragment_ids=("frag-001",), coverage=mask.sum() / mask.size, area=int(mask.sum()))

    image_paths = render_solution_gallery(template, [fragment], [solution], tmp_path / "images")
    report_path = tmp_path / "report.html"
    write_solution_report([solution], image_paths, report_path)

    assert image_paths[0].exists()
    html = report_path.read_text(encoding="utf-8")
    assert "Candidate 1" in html
    assert "images/solution_000.png" in html
    assert "frag-001" in html
