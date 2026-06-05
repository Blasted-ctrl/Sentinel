"""``sentinel`` command-line interface."""

from __future__ import annotations

import json
from datetime import date
from typing import Annotated

import typer

from sentinel.config import get_settings
from sentinel.db.base import get_engine, session_scope
from sentinel.db.schema import init_db
from sentinel.geo import BBox, parse_bbox
from sentinel.ingest.pipeline import build_pipeline
from sentinel.logging import configure_logging

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Sentinel — wildfire risk data ingestion and tooling.",
)


def _default_name(bbox: BBox) -> str:
    return (
        f"bbox_{bbox.min_lon}_{bbox.min_lat}_{bbox.max_lon}_{bbox.max_lat}".replace(
            "-", "m"
        )
    )


def _parse_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"{field} must be YYYY-MM-DD, got {value!r}") from exc


@app.command("init-db")
def init_db_command() -> None:
    """Enable PostGIS and create all tables."""
    settings = get_settings()
    configure_logging(settings.log_level, json=settings.log_json)
    init_db(get_engine())
    typer.echo("Schema initialised (PostGIS enabled, tables created).")


@app.command()
def ingest(
    region: Annotated[
        str,
        typer.Option(
            "--region", help="bbox as 'min_lon,min_lat,max_lon,max_lat' (WGS84)"
        ),
    ],
    start: Annotated[str, typer.Option("--start", help="start date YYYY-MM-DD")],
    end: Annotated[str, typer.Option("--end", help="end date YYYY-MM-DD")],
    name: Annotated[
        str | None, typer.Option("--name", help="region name (defaults to bbox)")
    ] = None,
    sources: Annotated[
        str, typer.Option("--sources", help="comma list: fire,weather,imagery")
    ] = "fire,weather,imagery",
    asset: Annotated[
        str, typer.Option("--asset", help="STAC asset to store (e.g. thumbnail)")
    ] = "thumbnail",
    tile_limit: Annotated[
        int, typer.Option("--tile-limit", help="max satellite scenes to fetch")
    ] = 10,
    init: Annotated[
        bool, typer.Option("--init/--no-init", help="create schema before ingesting")
    ] = False,
) -> None:
    """Fetch fire / weather / imagery for a region and store it in PostGIS + S3."""
    settings = get_settings()
    configure_logging(settings.log_level, json=settings.log_json)

    try:
        bbox = parse_bbox(region)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    start_date = _parse_date(start, "start")
    end_date = _parse_date(end, "end")
    if start_date > end_date:
        raise typer.BadParameter("start must not be after end")

    source_list = [s.strip() for s in sources.split(",") if s.strip()]
    if not source_list:
        raise typer.BadParameter("at least one source is required")

    region_name = name or _default_name(bbox)
    engine = get_engine()
    if init:
        init_db(engine)

    with session_scope(engine) as session:
        pipeline = build_pipeline(session, settings=settings, sources=source_list)
        summary = pipeline.run(
            region_name,
            bbox,
            start_date,
            end_date,
            sources=source_list,
            asset=asset,
            tile_limit=tile_limit,
        )

    typer.echo(json.dumps(summary.as_dict(), indent=2))


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
