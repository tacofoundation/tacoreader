"""Integration tests for tacoreader/_html.py - with real dataset fixtures."""

import pytest

from tacoreader._html import (
    build_html_repr,
    render_header,
    render_hierarchy_section,
    render_metadata_section,
    render_fields_section,
    render_graph,
)


class TestRenderHierarchySection:

    def test_flat_dataset_shows_root_level(self, ds_zip_flat):
        html = render_hierarchy_section(ds_zip_flat)

        assert "Level 0" in html
        assert "FILE" in html

    def test_nested_dataset_shows_child_levels(self, ds_zip_nested):
        html = render_hierarchy_section(ds_zip_nested)

        assert "Level 0" in html
        assert "Level 1" in html

    def test_deep_dataset_shows_multiple_levels(self, ds_folder_deep):
        html = render_hierarchy_section(ds_folder_deep)

        assert "Level 0" in html
        assert "levels" in html

    def test_nested_shows_ids_detail(self, ds_zip_nested):
        html = render_hierarchy_section(ds_zip_nested)

        assert "Show IDs" in html


class TestRenderGraph:

    def test_flat_dataset_renders_svg(self, ds_zip_flat):
        svg = render_graph(ds_zip_flat)

        assert "<svg" in svg
        assert "</svg>" in svg
        assert "<circle" in svg

    def test_nested_dataset_has_edges(self, ds_zip_nested):
        svg = render_graph(ds_zip_nested)

        assert "<line" in svg
        assert "arrowhead" in svg

    def test_folder_nested_renders(self, ds_folder_nested):
        svg = render_graph(ds_folder_nested)

        assert "<svg" in svg

    def test_deep_dataset_renders(self, ds_folder_deep):
        svg = render_graph(ds_folder_deep)

        assert "<svg" in svg
        assert "<circle" in svg

    def test_tacocat_renders(self, ds_tacocat):
        svg = render_graph(ds_tacocat)

        assert "<svg" in svg


class TestRenderHeader:

    def test_shows_sample_count(self, ds_zip_flat):
        html = render_header(ds_zip_flat)

        n_samples = ds_zip_flat.pit_schema.root["n"]
        assert str(n_samples) in html

    def test_shows_level_count(self, ds_zip_nested):
        html = render_header(ds_zip_nested)

        total_levels = ds_zip_nested.pit_schema.max_depth() + 1
        assert str(total_levels) in html

    def test_uses_title_if_present(self, ds_zip_flat):
        html = render_header(ds_zip_flat)

        expected = ds_zip_flat.title if ds_zip_flat.title else ds_zip_flat.id
        assert expected in html


class TestRenderMetadataSection:

    def test_includes_core_attributes(self, ds_zip_flat):
        html = render_metadata_section(ds_zip_flat)

        assert "id" in html
        assert "version" in html
        assert "description" in html

    def test_truncates_long_description(self, ds_zip_flat):
        original = ds_zip_flat.description
        ds_zip_flat.description = "x" * 200

        html = render_metadata_section(ds_zip_flat)

        assert "..." in html
        ds_zip_flat.description = original

    def test_includes_spatial_extent_when_present(self, ds_zip_flat):
        if "spatial" in ds_zip_flat.extent:
            html = render_metadata_section(ds_zip_flat)
            assert "spatial" in html

    def test_includes_temporal_when_present(self, ds_zip_nested):
        html = render_metadata_section(ds_zip_nested)

        if ds_zip_nested.extent.get("temporal"):
            assert "temporal" in html


class TestRenderFieldsSection:

    def test_shows_field_count(self, ds_zip_flat):
        html = render_fields_section(ds_zip_flat)

        assert "Fields" in html
        assert "total" in html

    def test_groups_by_level(self, ds_zip_nested):
        html = render_fields_section(ds_zip_nested)

        assert "level" in html.lower()


class TestBuildHtmlRepr:

    def test_contains_all_sections(self, ds_zip_flat):
        html = build_html_repr(ds_zip_flat)

        assert "taco-header" in html
        assert "taco-sections" in html
        assert "taco-graph" in html
        assert "<style>" in html

    def test_displays_dataset_identifier(self, ds_zip_flat):
        html = build_html_repr(ds_zip_flat)

        expected = ds_zip_flat.title if ds_zip_flat.title else ds_zip_flat.id
        assert expected in html

    def test_nested_shows_hierarchy_in_graph(self, ds_zip_nested):
        html = build_html_repr(ds_zip_nested)

        assert "Hierarchy" in html
        assert "<svg" in html
        assert "<line" in html

    def test_folder_format_renders(self, ds_folder_nested):
        html = build_html_repr(ds_folder_nested)

        assert "FOLDER" in html

    def test_deep_hierarchy_renders_completely(self, ds_folder_deep):
        html = build_html_repr(ds_folder_deep)

        assert "<svg" in html
        assert "taco-dataset-wrap" in html


class TestReprHtmlMethod:

    def test_flat_zip(self, ds_zip_flat):
        html = ds_zip_flat._repr_html_()

        assert isinstance(html, str)
        assert len(html) > 100
        assert "<div" in html

    def test_nested_zip(self, ds_zip_nested):
        html = ds_zip_nested._repr_html_()

        assert "<svg" in html

    def test_flat_folder(self, ds_folder_flat):
        html = ds_folder_flat._repr_html_()

        assert "taco-dataset-wrap" in html

    def test_nested_folder(self, ds_folder_nested):
        html = ds_folder_nested._repr_html_()

        assert "<svg" in html

    def test_deep_folder(self, ds_folder_deep):
        html = ds_folder_deep._repr_html_()

        assert "<svg" in html

    def test_tacocat(self, ds_tacocat):
        html = ds_tacocat._repr_html_()

        assert "taco-dataset-wrap" in html
        assert "<svg" in html