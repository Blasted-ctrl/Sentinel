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


@app.command("prepare-data")
def prepare_data(
    source: Annotated[
        str, typer.Option("--source", help="kaggle | zip | local")
    ] = "kaggle",
    path: Annotated[
        str | None, typer.Option("--path", help="zip file or folder (zip/local sources)")
    ] = None,
    dataset: Annotated[
        str, typer.Option("--dataset", help="Kaggle dataset slug")
    ] = "abdelghaniaaba/wildfire-prediction-dataset",
    dest: Annotated[
        str, typer.Option("--dest", help="extraction target for zip source")
    ] = "data/wildfire",
) -> None:
    """Download / locate the wildfire image dataset and print its root path."""
    settings = get_settings()
    configure_logging(settings.log_level, json=settings.log_json)
    from sentinel.data.download import resolve_wildfire_dataset  # lazy (heavy deps)

    root = resolve_wildfire_dataset(source, path=path, dataset=dataset, dest=dest)
    typer.echo(str(root))


@app.command("train-cnn")
def train_cnn_command(
    data_root: Annotated[
        str, typer.Option("--data-root", help="dataset root (from prepare-data)")
    ],
    output: Annotated[
        str, typer.Option("--output", help="directory for model + metrics.json")
    ] = "artifacts/cnn",
    epochs: Annotated[int, typer.Option("--epochs")] = 5,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 32,
    image_size: Annotated[int, typer.Option("--image-size")] = 96,
    limit: Annotated[
        int | None, typer.Option("--limit", help="cap total tiles (balanced) for CPU runs")
    ] = None,
    cell_size: Annotated[
        float, typer.Option("--cell-size", help="geospatial split grid size in degrees")
    ] = 0.5,
    lr: Annotated[float, typer.Option("--lr")] = 1e-3,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    pretrained: Annotated[
        bool, typer.Option("--pretrained/--no-pretrained")
    ] = True,
    unfreeze_layer4: Annotated[
        bool, typer.Option("--unfreeze-layer4/--freeze-layer4")
    ] = True,
    device: Annotated[str, typer.Option("--device", help="cpu | cuda")] = "cpu",
) -> None:
    """Train the satellite-imagery CNN and write model + metrics.json."""
    settings = get_settings()
    configure_logging(settings.log_level, json=settings.log_json)
    from pathlib import Path

    from sentinel.training.train_cnn import TrainConfig, train_cnn  # lazy (torch)

    cfg = TrainConfig(
        data_root=Path(data_root),
        output_dir=Path(output),
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
        limit=limit,
        cell_size=cell_size,
        lr=lr,
        seed=seed,
        pretrained=pretrained,
        unfreeze_layer4=unfreeze_layer4,
        device=device,
    )
    report = train_cnn(cfg)
    typer.echo(json.dumps(report["test"], indent=2))


@app.command("train-lstm")
def train_lstm_command(
    fod_sqlite: Annotated[
        str, typer.Option("--fod-sqlite", help="path to the FPA-FOD .sqlite file")
    ],
    output: Annotated[str, typer.Option("--output")] = "artifacts/lstm",
    region: Annotated[
        str | None,
        typer.Option("--region", help="bbox min_lon,min_lat,max_lon,max_lat"),
    ] = None,
    cell_size: Annotated[float, typer.Option("--cell-size")] = 1.0,
    start: Annotated[str, typer.Option("--start", help="YYYY-MM-DD")] = "2010-01-01",
    end: Annotated[str, typer.Option("--end", help="YYYY-MM-DD")] = "2015-12-31",
    lookback: Annotated[int, typer.Option("--lookback")] = 14,
    horizon: Annotated[int, typer.Option("--horizon")] = 7,
    epochs: Annotated[int, typer.Option("--epochs")] = 8,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Train the climate LSTM and write model + metrics.json."""
    settings = get_settings()
    configure_logging(settings.log_level, json=settings.log_json)
    from pathlib import Path

    from sentinel.training.train_lstm import (  # lazy (tensorflow)
        DEFAULT_BBOX,
        LstmTrainConfig,
        train_lstm,
    )

    bbox = DEFAULT_BBOX
    if region is not None:
        try:
            bbox = parse_bbox(region)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    cfg = LstmTrainConfig(
        fod_sqlite=Path(fod_sqlite),
        output_dir=Path(output),
        bbox=bbox,
        cell_size=cell_size,
        start=_parse_date(start, "start"),
        end=_parse_date(end, "end"),
        lookback=lookback,
        horizon=horizon,
        epochs=epochs,
        seed=seed,
    )
    report = train_lstm(cfg)
    typer.echo(json.dumps(report["test"], indent=2))


@app.command("build-ensemble")
def build_ensemble_command(
    lstm_dir: Annotated[
        str, typer.Option("--lstm-dir", help="LSTM output dir (with ensemble_inputs.npz)")
    ],
    cnn_checkpoint: Annotated[
        str, typer.Option("--cnn-checkpoint", help="path to the CNN .pt checkpoint")
    ],
    output: Annotated[str, typer.Option("--output")] = "artifacts/ensemble",
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Fit + evaluate the CNN+LSTM ensemble and write metrics.json."""
    settings = get_settings()
    configure_logging(settings.log_level, json=settings.log_json)
    from pathlib import Path

    from sentinel.training.build_ensemble import (  # lazy (torch)
        EnsembleConfig,
        build_ensemble,
    )

    cfg = EnsembleConfig(
        lstm_dir=Path(lstm_dir),
        cnn_checkpoint=Path(cnn_checkpoint),
        output_dir=Path(output),
        seed=seed,
    )
    report = build_ensemble(cfg)
    typer.echo(json.dumps(report["test"]["ensemble"], indent=2))


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
