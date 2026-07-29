from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.pid_extraction import yolo_detector
from src.pid_extraction.pipeline import _extract_page_graph, _merge_shapes
from src.pid_extraction.shape_detection import DetectedShape


def _shape(kind):
    return DetectedShape(shape_id=0, kind=kind, bbox=(0, 0, 10, 10), center=(5, 5), contour=np.empty((0, 1, 2), dtype=np.int32))


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_merge_shapes_no_yolo_shapes_returns_empty_overrides():
    shapes, overrides = _merge_shapes([_shape("circle")], [_shape("bowtie")])
    assert len(shapes) == 2
    assert overrides == {}


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_merge_shapes_yolo_shapes_get_component_type_overrides():
    """YOLO shapes are appended last and re-numbered into the shared shape_id
    space; each gets an override entry keyed by its *new* id, not shape_id=0
    that every input DetectedShape carries before merging."""
    raster = [_shape("circle")]
    vector = [_shape("bowtie")]
    yolo = [_shape("gate_valve"), _shape("pressure_transmitter")]

    shapes, overrides = _merge_shapes(raster, vector, yolo_shapes=yolo)

    assert len(shapes) == 4
    assert [s.shape_id for s in shapes] == [0, 1, 2, 3]
    # vector shapes are ordered before raster shapes, then yolo shapes last
    assert shapes[0].kind == "bowtie"
    assert shapes[1].kind == "circle"
    assert shapes[2].kind == "gate_valve"
    assert shapes[3].kind == "pressure_transmitter"
    assert overrides == {2: "valve", 3: "instrument"}


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_merge_shapes_yolo_only_unmapped_class_falls_back_to_unknown_edge_case():
    shapes, overrides = _merge_shapes([], yolo_shapes=[_shape("manhole")])
    assert overrides == {0: "unknown"}


@pytest.mark.regression
@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_extract_page_graph_picks_up_runtime_default_weights_path_change(monkeypatch, tmp_path):
    """Regression: Python binds a function's default argument values once at
    definition time. Reassigning yolo_detector.DEFAULT_WEIGHTS_PATH at runtime
    used to be silently ignored by any caller that didn't pass weights_path
    explicitly (including pipeline.py's own yolo call) — every "checkpoint
    comparison" done this way kept re-testing whatever model was default when
    the module was first imported, regardless of later reassignment. Confirms
    the fix: _extract_page_graph now reads the module attribute live."""
    new_default = tmp_path / "new_default.pt"
    monkeypatch.setattr(yolo_detector, "DEFAULT_WEIGHTS_PATH", new_default)

    with (
        patch("src.pid_extraction.pipeline.weights_available", return_value=True) as mock_available,
        patch("src.pid_extraction.pipeline.detect_yolo_symbols", return_value=[]) as mock_detect,
        patch("src.pid_extraction.pipeline.detect_shapes", return_value=[]),
    ):
        _extract_page_graph(np.zeros((10, 10, 3), dtype=np.uint8), use_yolo=True)

    mock_available.assert_called_once_with(new_default)
    mock_detect.assert_called_once()
    assert mock_detect.call_args[0][1] == new_default


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_extract_page_graph_explicit_weights_path_overrides_default():
    explicit_path = Path("/explicit/weights.pt")

    with (
        patch("src.pid_extraction.pipeline.weights_available", return_value=True) as mock_available,
        patch("src.pid_extraction.pipeline.detect_yolo_symbols", return_value=[]) as mock_detect,
        patch("src.pid_extraction.pipeline.detect_shapes", return_value=[]),
    ):
        _extract_page_graph(np.zeros((10, 10, 3), dtype=np.uint8), use_yolo=True, yolo_weights_path=explicit_path)

    mock_available.assert_called_once_with(explicit_path)
    assert mock_detect.call_args[0][1] == explicit_path


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_symbol_type_assist_fills_component_type_for_unknown_shape():
    """Phase 1 wiring: a shape with no tag-resolved type, no YOLO override,
    and a kind outside SHAPE_TO_TYPE's vocabulary (kind="unknown", so it would
    otherwise reach the graph as component_type "unknown") gets classified by
    the local vision model instead when symbol_type_assist=True."""
    shape = _shape("unknown")

    with (
        patch("src.pid_extraction.pipeline.detect_shapes", return_value=[shape]),
        patch("src.pid_extraction.pipeline.extract_tag", return_value=(None, "")),
        patch("src.pid_extraction.pipeline.classify_symbol_type_with_llm", return_value=("valve", "gate_valve", "valve, gate_valve")) as mock_classify,
    ):
        graph = _extract_page_graph(np.zeros((10, 10, 3), dtype=np.uint8), symbol_type_assist=True)

    mock_classify.assert_called_once()
    node_data = next(iter(graph.nodes(data=True)))[1]
    assert node_data["component_type"] == "gate_valve"


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_symbol_type_assist_never_overrides_tag_resolved_type_edge_case():
    """A shape with a real, tag-resolved type must never be sent to the vision
    model at all -- symbol_type_assist only fills gaps, it doesn't second-guess
    a cheaper/more-certain path."""
    shape = _shape("circle")  # circle already maps to "instrument" via SHAPE_TO_TYPE

    with (
        patch("src.pid_extraction.pipeline.detect_shapes", return_value=[shape]),
        patch("src.pid_extraction.pipeline.extract_tag", return_value=("PI-101", "PI-101")),
        patch("src.pid_extraction.pipeline.classify_symbol_type_with_llm") as mock_classify,
    ):
        graph = _extract_page_graph(np.zeros((10, 10, 3), dtype=np.uint8), symbol_type_assist=True)

    mock_classify.assert_not_called()
    node_data = graph.nodes["PI-101"]
    assert node_data["component_type"] == "pressure_indicator"


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_symbol_type_assist_off_by_default_edge_case():
    """symbol_type_assist=False (the default) must never call the vision
    model, matching llm_ocr_assist's own opt-in-only contract."""
    shape = _shape("unknown")

    with (
        patch("src.pid_extraction.pipeline.detect_shapes", return_value=[shape]),
        patch("src.pid_extraction.pipeline.extract_tag", return_value=(None, "")),
        patch("src.pid_extraction.pipeline.classify_symbol_type_with_llm") as mock_classify,
    ):
        graph = _extract_page_graph(np.zeros((10, 10, 3), dtype=np.uint8))

    mock_classify.assert_not_called()
    node_data = next(iter(graph.nodes(data=True)))[1]
    assert node_data["component_type"] == "unknown"


@pytest.mark.unit
@pytest.mark.failure_path
@pytest.mark.authored_claude_sonnet
def test_symbol_type_assist_unresolved_classification_leaves_shape_unknown_failure_path():
    """When the model can't confidently classify a shape either (both
    coarse/subtype come back None), the shape stays "unknown" -- same as if
    the flag were off, not a crash and not a fabricated guess."""
    shape = _shape("unknown")

    with (
        patch("src.pid_extraction.pipeline.detect_shapes", return_value=[shape]),
        patch("src.pid_extraction.pipeline.extract_tag", return_value=(None, "")),
        patch("src.pid_extraction.pipeline.classify_symbol_type_with_llm", return_value=(None, None, "unknown")),
    ):
        graph = _extract_page_graph(np.zeros((10, 10, 3), dtype=np.uint8), symbol_type_assist=True)

    node_data = next(iter(graph.nodes(data=True)))[1]
    assert node_data["component_type"] == "unknown"


def _large_shape(kind, w=101, h=437):
    """A shape sized like a real vessel (F-715A: 101x437px) -- large enough
    to trip the oversized-crop guard, unlike the 10x10 default test shape."""
    return DetectedShape(shape_id=0, kind=kind, bbox=(0, 0, w, h), center=(w // 2, h // 2), contour=np.empty((0, 1, 2), dtype=np.int32))


@pytest.mark.regression
@pytest.mark.unit
@pytest.mark.failure_path
@pytest.mark.authored_claude_sonnet
def test_llm_ocr_assist_skips_oversized_shapes_regression():
    """Real bug found live this session: llm_ocr_assist misread F-715A's and
    F-715B's crops as neighboring tags ("FT-101", "MV-715B") because
    _crop_label_region's margin-based crop balloons to the shape's full bbox
    plus margin -- fine for small instrument/valve shapes, but for a 437px-tall
    vessel it sweeps in unrelated nearby symbols' tags. Those wrong reads then
    silently overrode the correct "tank"->"equipment" type via graph_builder's
    tag-priority rule (measured: page 0 equipment coverage 2/2 -> 0/2 with
    llm_ocr_assist on). Fix: skip LLM tag-reading for any shape whose bbox
    exceeds _MAX_LABEL_CROP_SHAPE_DIM_PX -- its type is already known reliably
    from geometry in that case, so a guessed tag can only hurt."""
    shape = _large_shape("rectangle")  # SHAPE_TO_TYPE["rectangle"] == "tank" -- already known

    with (
        patch("src.pid_extraction.pipeline.detect_shapes", return_value=[shape]),
        patch("src.pid_extraction.pipeline.extract_tag", return_value=(None, "")),
        patch("src.pid_extraction.pipeline.read_tag_with_llm") as mock_llm,
    ):
        graph = _extract_page_graph(np.zeros((500, 500, 3), dtype=np.uint8), llm_ocr_assist=True)

    mock_llm.assert_not_called()
    node_data = next(iter(graph.nodes(data=True)))[1]
    assert node_data["component_type"] == "tank"


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_llm_ocr_assist_still_runs_for_normal_sized_shapes_happy_path():
    """Confirms the size guard doesn't over-trigger -- ordinary instrument/
    valve-sized shapes (well under the threshold) still get LLM tag reading."""
    shape = _shape("circle")  # 10x10, far under the threshold

    with (
        patch("src.pid_extraction.pipeline.detect_shapes", return_value=[shape]),
        patch("src.pid_extraction.pipeline.extract_tag", return_value=(None, "")),
        patch("src.pid_extraction.pipeline.read_tag_with_llm", return_value=("PI-101", "PI-101")) as mock_llm,
    ):
        graph = _extract_page_graph(np.zeros((100, 100, 3), dtype=np.uint8), llm_ocr_assist=True)

    mock_llm.assert_called_once()
    assert "PI-101" in graph.nodes


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_symbol_type_assist_also_skips_oversized_unknown_shapes_edge_case():
    shape = _large_shape("unknown")

    with (
        patch("src.pid_extraction.pipeline.detect_shapes", return_value=[shape]),
        patch("src.pid_extraction.pipeline.extract_tag", return_value=(None, "")),
        patch("src.pid_extraction.pipeline.classify_symbol_type_with_llm") as mock_classify,
    ):
        graph = _extract_page_graph(np.zeros((500, 500, 3), dtype=np.uint8), symbol_type_assist=True)

    mock_classify.assert_not_called()
    node_data = next(iter(graph.nodes(data=True)))[1]
    assert node_data["component_type"] == "unknown"


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_equipment_label_discovery_adds_candidate_as_equipment_happy_path():
    """A label-discovered candidate (e.g. the real P-745 pump gap -- no
    detector produces any shape near it) is merged into the shape list and
    typed as equipment directly, since the discovery mechanism only fires on
    equipment-prefix tags to begin with."""
    candidate = _shape("equipment_label_region")

    with (
        patch("src.pid_extraction.pipeline.detect_shapes", return_value=[]),
        patch("src.pid_extraction.pipeline.extract_tag", return_value=(None, "")),
        patch("src.pid_extraction.pipeline.find_equipment_label_candidates", return_value=[candidate]) as mock_find,
    ):
        graph = _extract_page_graph(np.zeros((10, 10, 3), dtype=np.uint8), equipment_label_discovery=True)

    mock_find.assert_called_once()
    node_data = next(iter(graph.nodes(data=True)))[1]
    assert node_data["component_type"] == "equipment"


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_equipment_label_discovery_off_by_default_edge_case():
    with (
        patch("src.pid_extraction.pipeline.detect_shapes", return_value=[]),
        patch("src.pid_extraction.pipeline.extract_tag", return_value=(None, "")),
        patch("src.pid_extraction.pipeline.find_equipment_label_candidates") as mock_find,
    ):
        _extract_page_graph(np.zeros((10, 10, 3), dtype=np.uint8))

    mock_find.assert_not_called()


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_equipment_label_discovery_checked_only_against_vessel_shapes_edge_case():
    """find_equipment_label_candidates must be called with vessel_shapes
    (the vector-signature equipment) for its exclusion check, not the full
    merged shape list -- nearby valves/instruments must not suppress a real
    discovery (this was the actual real bug caught while building this: a
    naive exclusion check against the wrong shape set wrongly re-suppressed
    or wrongly allowed candidates)."""
    vessel = _shape("rectangle")

    with (
        patch("src.pid_extraction.pipeline.detect_shapes", return_value=[]),
        patch("src.pid_extraction.pipeline.extract_circle_symbols", return_value=[]),
        patch("src.pid_extraction.pipeline.extract_valve_symbols", return_value=[]),
        patch("src.pid_extraction.pipeline.extract_vessel_symbols", return_value=[vessel]),
        patch("src.pid_extraction.pipeline.extract_line_segments", return_value=[]),
        patch("src.pid_extraction.pipeline.extract_tag", return_value=(None, "")),
        patch("src.pid_extraction.pipeline.find_equipment_label_candidates", return_value=[]) as mock_find,
    ):
        _extract_page_graph(
            np.zeros((10, 10, 3), dtype=np.uint8), pdf_path="fake.pdf", equipment_label_discovery=True,
        )

    call_kwargs = mock_find.call_args
    passed_shapes = call_kwargs.args[1] if len(call_kwargs.args) > 1 else call_kwargs.kwargs.get("existing_equipment_shapes")
    assert passed_shapes == [vessel]


@pytest.mark.regression
@pytest.mark.unit
@pytest.mark.failure_path
@pytest.mark.authored_claude_sonnet
def test_anomalous_oversized_valve_shape_filtered_out_regression():
    """Regression for the real gap found diagnosing the tag-recall gap
    directly: deterministic OCR against every valve-typed shape on page 0
    turned up several with bboxes like 48x495px -- nearly 500px tall, vs.
    every real valve/instrument bbox hand-verified this session topping out
    around 105px. These are almost certainly malformed/mis-scaled YOLO
    detections (confirmed: pure OCR garbage, not near-misses), inflating the
    already-disclosed raw over-detection counts. Must be dropped entirely,
    not just skipped for tag-reading."""
    normal_valve = _shape("bowtie")  # 10x10, well under the threshold
    anomalous_valve = DetectedShape(
        shape_id=1, kind="bowtie", bbox=(0, 0, 48, 495), center=(24, 247),
        contour=np.empty((0, 1, 2), dtype=np.int32),
    )

    with (
        patch("src.pid_extraction.pipeline.detect_shapes", return_value=[normal_valve, anomalous_valve]),
        patch("src.pid_extraction.pipeline.extract_tag", return_value=(None, "")),
    ):
        graph = _extract_page_graph(np.zeros((600, 600, 3), dtype=np.uint8))

    assert graph.number_of_nodes() == 1  # only the normal valve survives
    (node_data,) = [d for _, d in graph.nodes(data=True)]
    assert node_data["bbox"] == [0, 0, 10, 10]


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_anomalous_size_filter_exempts_equipment_edge_case():
    """The size filter must not touch equipment -- large equipment (vessels,
    discovered labels) is legitimate; only valve/instrument-coarse shapes
    get filtered."""
    large_vessel = _large_shape("rectangle")  # 101x437px, real F-715A scale

    with (
        patch("src.pid_extraction.pipeline.detect_shapes", return_value=[large_vessel]),
        patch("src.pid_extraction.pipeline.extract_tag", return_value=(None, "")),
    ):
        graph = _extract_page_graph(np.zeros((600, 600, 3), dtype=np.uint8))

    assert graph.number_of_nodes() == 1
    (node_data,) = [d for _, d in graph.nodes(data=True)]
    assert node_data["component_type"] == "tank"


@pytest.mark.regression
@pytest.mark.unit
@pytest.mark.failure_path
@pytest.mark.authored_claude_sonnet
def test_equipment_label_region_skips_narrow_extract_tag_regression():
    """Regression for a real, confirmed-live bug found diagnosing page 2's
    AC-746/E-742 false negatives: extract_tag's rotation+PSM binarized
    single-tag pipeline is built for tight crops, not the large (440px)
    multi-line context regions equipment_label_discovery produces -- on the
    real document it confidently misread "E-742" as "E-42" (3 chars away
    from correct). Because extract_tag returns *something* non-None, that
    wrong tag silently won over the correct wide-crop word search below
    (which only runs when the tag is still None). equipment_label_region
    candidates must skip extract_tag entirely so the wide-crop search always
    gets the chance to run."""
    candidate = _large_shape("equipment_label_region", w=440, h=440)  # real discovery-candidate scale

    with (
        patch("src.pid_extraction.pipeline.detect_shapes", return_value=[]),
        patch("src.pid_extraction.pipeline.extract_tag", return_value=("E-42", "E-42")) as mock_extract_tag,
        patch("src.pid_extraction.pipeline.find_equipment_label_candidates", return_value=[candidate]),
        patch("src.pid_extraction.pipeline.find_tag_for_known_equipment_shape", return_value="E-742") as mock_find_tag,
    ):
        graph = _extract_page_graph(np.zeros((10, 10, 3), dtype=np.uint8), equipment_label_discovery=True)

    mock_extract_tag.assert_not_called()
    mock_find_tag.assert_called_once()
    node_data = next(iter(graph.nodes(data=True)))[1]
    assert node_data["tag"] == "E-742"


@pytest.mark.unit
@pytest.mark.happy_path
@pytest.mark.authored_claude_sonnet
def test_equipment_label_discovery_finds_tag_for_known_oversized_shape_happy_path():
    """Closes the gap the size guard itself created: F-715A/B-style shapes
    never get their own tag read via the normal path (by design, to prevent
    the oversized-crop misread bug) -- this targeted search is the intended
    way their real tag text becomes reachable at all."""
    vessel = _large_shape("rectangle")  # 101x437px, tag-reading normally blocked

    with (
        patch("src.pid_extraction.pipeline.detect_shapes", return_value=[]),
        patch("src.pid_extraction.pipeline.extract_circle_symbols", return_value=[]),
        patch("src.pid_extraction.pipeline.extract_valve_symbols", return_value=[]),
        patch("src.pid_extraction.pipeline.extract_vessel_symbols", return_value=[vessel]),
        patch("src.pid_extraction.pipeline.extract_line_segments", return_value=[]),
        patch("src.pid_extraction.pipeline.extract_tag", return_value=(None, "")),
        patch("src.pid_extraction.pipeline.find_equipment_label_candidates", return_value=[]),
        patch("src.pid_extraction.pipeline.find_tag_for_known_equipment_shape", return_value="F-715A") as mock_find_tag,
    ):
        graph = _extract_page_graph(
            np.zeros((600, 600, 3), dtype=np.uint8), pdf_path="fake.pdf", equipment_label_discovery=True,
        )

    mock_find_tag.assert_called_once()
    assert "F-715A" in graph.nodes
    assert graph.nodes["F-715A"]["tag_confidence"] == "equipment_label_discovery"


@pytest.mark.unit
@pytest.mark.edge_case
@pytest.mark.authored_claude_sonnet
def test_known_equipment_tag_search_off_by_default_edge_case():
    vessel = _large_shape("rectangle")

    with (
        patch("src.pid_extraction.pipeline.detect_shapes", return_value=[]),
        patch("src.pid_extraction.pipeline.extract_circle_symbols", return_value=[]),
        patch("src.pid_extraction.pipeline.extract_valve_symbols", return_value=[]),
        patch("src.pid_extraction.pipeline.extract_vessel_symbols", return_value=[vessel]),
        patch("src.pid_extraction.pipeline.extract_line_segments", return_value=[]),
        patch("src.pid_extraction.pipeline.extract_tag", return_value=(None, "")),
        patch("src.pid_extraction.pipeline.find_tag_for_known_equipment_shape") as mock_find_tag,
    ):
        _extract_page_graph(np.zeros((600, 600, 3), dtype=np.uint8), pdf_path="fake.pdf")

    mock_find_tag.assert_not_called()
