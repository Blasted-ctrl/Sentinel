"""Tests for bbox parsing and geometry helpers."""

from __future__ import annotations

import pytest

from sentinel.geo import BBox, parse_bbox


def test_parse_bbox_valid() -> None:
    bbox = parse_bbox("-120.5, 38.5, -120.0, 39.0")
    assert bbox == BBox(-120.5, 38.5, -120.0, 39.0)


def test_centroid() -> None:
    bbox = BBox(-120.5, 38.5, -120.0, 39.0)
    lon, lat = bbox.centroid
    assert lon == pytest.approx(-120.25)
    assert lat == pytest.approx(38.75)


def test_to_wkt_is_polygon() -> None:
    wkt = BBox(-1.0, -1.0, 1.0, 1.0).to_wkt()
    assert wkt.startswith("POLYGON")
    assert "-1 -1" in wkt or "-1.0 -1.0" in wkt


def test_as_list_order() -> None:
    assert BBox(1.0, 2.0, 3.0, 4.0).as_list() == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize(
    "value",
    [
        "1,2,3",  # too few
        "1,2,3,4,5",  # too many
        "a,b,c,d",  # non-numeric
        "200,0,201,1",  # lon out of range
        "0,-100,1,100",  # lat out of range
        "3,2,1,4",  # min_lon >= max_lon
        "1,4,3,2",  # min_lat >= max_lat
    ],
)
def test_parse_bbox_invalid(value: str) -> None:
    with pytest.raises(ValueError):
        parse_bbox(value)
