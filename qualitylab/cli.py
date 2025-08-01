import click
from pathlib import Path
import pandas as pd

from .spreadsheets import read_production_data, read_downtime_data
from .feature_engineering import add_recent_history
from .build_time import train_build_time_model
from .build_quantity import train_build_quantity_model
from .defects import train_defect_model
from .paths import get_data_dir



@click.group()
def cli():
    pass

@cli.command("ingest")
@click.argument("files", nargs=-1, type=click.Path(exists=True))
def ingest(files):
    df = read_production_data([Path(f) for f in files])
    out_dir = get_data_dir()
    df.to_parquet(out_dir / "production.parquet")
    click.echo("✅ Ingested")

@cli.command("ingest-downtime")
@click.argument("files", nargs=-1, type=click.Path(exists=True))
def ingest_downtime(files):
    df = read_downtime_data([Path(f) for f in files])
    out_dir = get_data_dir()
    df.to_parquet(out_dir / "downtime.parquet")
    click.echo("✅ Downtime ingested")

def _train_build_time() -> None:
    """Train the build time model using ingested production data."""
    data_path = get_data_dir() / "production.parquet"
    df = pd.read_parquet(data_path)
    train_build_time_model(df)


@cli.command("train-build-time")
def train_build_time() -> None:
    """CLI wrapper around :func:`_train_build_time`."""
    _train_build_time()
    click.echo("✅ Build-time model trained")

def _train_defects() -> None:
    """Train the defect prediction model using ingested production data."""
    data_path = get_data_dir() / "production.parquet"
    df = pd.read_parquet(data_path)
    train_defect_model(df)


@cli.command("train-defects")
def train_defects() -> None:
    """CLI wrapper around :func:`_train_defects`."""
    _train_defects()
    click.echo("✅ Defect model trained")

@cli.command("train-all")
def train_all():
    """Train all models in sequence."""
    _train_build_time()
    _train_defects()
    _train_build_quantity()


def _train_build_quantity() -> None:
    """Train the build quantity model using production and downtime data."""
    data_dir = get_data_dir()
    prod_path = data_dir / 'production.parquet'
    down_path = data_dir / 'downtime.parquet'
    if not prod_path.exists():
        raise click.ClickException('Production data not found. Run "qualitylab ingest" first.')
    if not down_path.exists():
        raise click.ClickException('Downtime data not found. Run "qualitylab ingest-downtime" first.')
    df_prod = pd.read_parquet(prod_path)
    df_prod = add_recent_history(df_prod)
    df_down = pd.read_parquet(down_path)
    train_build_quantity_model(df_prod, df_down)


@cli.command('train-build-quantity')
def train_build_quantity() -> None:
    """CLI wrapper around :func:`_train_build_quantity`."""
    _train_build_quantity()
    click.echo("✅ Build-quantity model trained")

if __name__ == "__main__":
    cli()
