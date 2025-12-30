"""Unit tests for tacoreader/_html.py - no I/O, no fixtures."""

import pytest

from tacoreader._html import (
    get_node_color,
    _determine_children_to_show,
    _adjust_node_positions,
    _create_child_nodes,
    _expand_hierarchy,
    _render_svg_elements,
    _create_root_node,
    LEFT_MARGIN,
    MAX_CHILDREN_FULL,
    START_X,
    START_Y,
)


class TestDetermineChildrenToShow:

    def test_shows_all_when_under_limit(self):
        indices, ellipsis, hidden = _determine_children_to_show(3)
        assert indices == [0, 1, 2]
        assert not ellipsis
        assert hidden == 0

    def test_shows_all_at_exact_limit(self):
        indices, ellipsis, hidden = _determine_children_to_show(MAX_CHILDREN_FULL)
        assert len(indices) == MAX_CHILDREN_FULL
        assert not ellipsis
        assert hidden == 0

    def test_truncates_when_over_limit(self):
        indices, ellipsis, hidden = _determine_children_to_show(10)
        assert indices == [0, 1, 9]
        assert ellipsis
        assert hidden == 7

    def test_single_child(self):
        indices, ellipsis, hidden = _determine_children_to_show(1)
        assert indices == [0]
        assert not ellipsis


class TestAdjustNodePositions:

    def test_empty_lists_noop(self):
        nodes, edges, texts = [], [], []
        _adjust_node_positions(nodes, edges, texts)
        assert nodes == []

    def test_no_shift_when_within_margin(self):
        nodes = [{"x": LEFT_MARGIN + 50, "y": 100}]
        edges = [{"from_x": LEFT_MARGIN + 50, "from_y": 50, "to_x": LEFT_MARGIN + 50, "to_y": 100}]
        texts = [{"x": LEFT_MARGIN + 50, "y": 75, "text": "..."}]

        original_x = nodes[0]["x"]
        _adjust_node_positions(nodes, edges, texts)

        assert nodes[0]["x"] == original_x

    def test_shifts_all_elements_when_exceeds_margin(self):
        nodes = [{"x": 5, "y": 100}, {"x": 50, "y": 100}]
        edges = [{"from_x": 5, "from_y": 50, "to_x": 50, "to_y": 100}]
        texts = [{"x": 25, "y": 75, "text": "..."}]

        _adjust_node_positions(nodes, edges, texts)

        assert all(n["x"] >= LEFT_MARGIN for n in nodes)
        assert edges[0]["from_x"] >= LEFT_MARGIN
        assert edges[0]["to_x"] >= LEFT_MARGIN
        assert texts[0]["x"] >= LEFT_MARGIN


class TestRenderSvgElements:

    def test_empty_graph_has_default_width(self):
        svg = _render_svg_elements([], [], [], 100)
        assert "<svg" in svg
        assert "</svg>" in svg
        assert 'width="250"' in svg

    def test_nodes_render_as_circles(self):
        nodes = [{"x": 100, "y": 50, "color": "#FFB6C1", "label": "root", "type": "TACO"}]
        svg = _render_svg_elements(nodes, [], [], 100)

        assert "<circle" in svg
        assert 'cx="100"' in svg
        assert 'fill="#FFB6C1"' in svg

    def test_edges_render_with_arrowhead(self):
        edges = [{"from_x": 100, "from_y": 50, "to_x": 100, "to_y": 150}]
        svg = _render_svg_elements([], edges, [], 200)

        assert "<line" in svg
        assert "marker-end" in svg
        assert "arrowhead" in svg

    def test_zero_length_edge_skipped(self):
        edges = [{"from_x": 100, "from_y": 50, "to_x": 100, "to_y": 50}]
        svg = _render_svg_elements([], edges, [], 100)
        assert "<line" not in svg

    def test_texts_render_with_custom_size(self):
        texts = [{"x": 100, "y": 75, "text": "...", "size": "10"}]
        svg = _render_svg_elements([], [], texts, 100)

        assert "<text" in svg
        assert "..." in svg
        assert 'font-size="10"' in svg

    def test_texts_default_size(self):
        texts = [{"x": 100, "y": 75, "text": "test"}]
        svg = _render_svg_elements([], [], texts, 100)
        assert 'font-size="12"' in svg


class TestGetNodeColor:

    def test_taco_color(self):
        assert get_node_color("TACO") == "#FFB6C1"

    def test_folder_color(self):
        assert get_node_color("FOLDER") == "#90EE90"

    def test_file_color(self):
        assert get_node_color("FILE") == "#DDA0DD"

    def test_unknown_defaults_to_taco(self):
        assert get_node_color("WEIRD") == "#FFB6C1"


class TestCreateRootNode:

    def test_root_node_properties(self):
        node = _create_root_node()

        assert node["id"] == "root"
        assert node["type"] == "TACO"
        assert node["level"] == -1
        assert node["x"] == START_X
        assert node["y"] == START_Y


class TestCreateChildNodes:

    def test_creates_nodes_for_all_indices(self):
        parent = {"id": "p", "x": 100, "y": 50, "type": "FOLDER"}
        nodes, edges, texts = [], [], []

        _create_child_nodes(
            parent,
            types=["FILE", "FILE"],
            ids=["a", "b"],
            current_depth=1,
            current_y=100,
            indices_to_show=[0, 1],
            show_ellipsis=False,
            hidden_count=0,
            nodes=nodes,
            edges=edges,
            texts=texts,
        )

        assert len(nodes) == 2
        assert len(edges) == 2
        assert nodes[0]["label"] == "a"
        assert nodes[1]["label"] == "b"

    def test_returns_first_folder_for_expansion(self):
        parent = {"id": "p", "x": 100, "y": 50, "type": "FOLDER"}
        nodes, edges, texts = [], [], []

        result = _create_child_nodes(
            parent,
            types=["FILE", "FOLDER", "FILE"],
            ids=["a", "b", "c"],
            current_depth=1,
            current_y=100,
            indices_to_show=[0, 1, 2],
            show_ellipsis=False,
            hidden_count=0,
            nodes=nodes,
            edges=edges,
            texts=texts,
        )

        assert result is not None
        assert result["type"] == "FOLDER"
        assert result["label"] == "b"

    def test_returns_none_when_no_folders(self):
        parent = {"id": "p", "x": 100, "y": 50, "type": "FOLDER"}
        nodes, edges, texts = [], [], []

        result = _create_child_nodes(
            parent,
            types=["FILE", "FILE"],
            ids=["a", "b"],
            current_depth=1,
            current_y=100,
            indices_to_show=[0, 1],
            show_ellipsis=False,
            hidden_count=0,
            nodes=nodes,
            edges=edges,
            texts=texts,
        )

        assert result is None

    def test_adds_ellipsis_text_when_truncated(self):
        parent = {"id": "p", "x": 100, "y": 50, "type": "FOLDER"}
        nodes, edges, texts = [], [], []

        _create_child_nodes(
            parent,
            types=["FILE"] * 10,
            ids=[f"f{i}" for i in range(10)],
            current_depth=1,
            current_y=100,
            indices_to_show=[0, 1, 9],
            show_ellipsis=True,
            hidden_count=7,
            nodes=nodes,
            edges=edges,
            texts=texts,
        )

        ellipsis_texts = [t for t in texts if "..." in t["text"]]
        assert len(ellipsis_texts) >= 1
        assert "(+7)" in ellipsis_texts[0]["text"]

    def test_truncates_long_labels(self):
        parent = {"id": "p", "x": 100, "y": 50, "type": "FOLDER"}
        nodes, edges, texts = [], [], []

        _create_child_nodes(
            parent,
            types=["FILE"],
            ids=["this_is_a_very_long_identifier"],
            current_depth=1,
            current_y=100,
            indices_to_show=[0],
            show_ellipsis=False,
            hidden_count=0,
            nodes=nodes,
            edges=edges,
            texts=texts,
        )

        assert len(nodes[0]["label"]) == 10


class TestExpandHierarchy:

    def test_empty_hierarchy_returns_start_y(self):
        parent = {"id": "p", "x": 100, "y": 50}
        nodes, edges, texts = [], [], []

        final_y = _expand_hierarchy(parent, {}, 1, 100, nodes, edges, texts)

        assert final_y == 100
        assert len(nodes) == 0

    def test_expands_single_level(self):
        parent = {"id": "p", "x": 100, "y": 50}
        hierarchy = {"1": [{"type": ["FILE", "FILE"], "id": ["a", "b"], "n": 2}]}
        nodes, edges, texts = [], [], []

        final_y = _expand_hierarchy(parent, hierarchy, 1, 100, nodes, edges, texts)

        assert len(nodes) == 2
        assert final_y > 100

    def test_expands_nested_folders(self):
        parent = {"id": "p", "x": 100, "y": 50}
        hierarchy = {
            "1": [{"type": ["FOLDER"], "id": ["sub"], "n": 1}],
            "2": [{"type": ["FILE"], "id": ["leaf"], "n": 1}],
        }
        nodes, edges, texts = [], [], []

        _expand_hierarchy(parent, hierarchy, 1, 100, nodes, edges, texts)

        assert len(nodes) == 2
        levels = {n["level"] for n in nodes}
        assert levels == {1, 2}

    def test_stops_at_file_level(self):
        parent = {"id": "p", "x": 100, "y": 50}
        hierarchy = {
            "1": [{"type": ["FILE"], "id": ["leaf"], "n": 1}],
            "2": [{"type": ["FILE"], "id": ["unreachable"], "n": 1}],
        }
        nodes, edges, texts = [], [], []

        _expand_hierarchy(parent, hierarchy, 1, 100, nodes, edges, texts)

        assert len(nodes) == 1
        assert nodes[0]["level"] == 1

    def test_stops_on_empty_patterns(self):
        parent = {"id": "p", "x": 100, "y": 50}
        hierarchy = {"1": []}
        nodes, edges, texts = [], [], []

        final_y = _expand_hierarchy(parent, hierarchy, 1, 100, nodes, edges, texts)

        assert final_y == 100
        assert len(nodes) == 0