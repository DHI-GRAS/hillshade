import glob
import pathlib
import concurrent.futures
from typing import Any, List, Tuple, Generator
import xml.etree.ElementTree as ET
import numpy as np
import rasterio
import tqdm
import click
from .hillshade import hillshade


class GlobbityGlob(click.ParamType):
    """Expands a glob pattern to Path objects"""
    name = 'glob'

    def convert(self, value: str, *args: Any) -> List[pathlib.Path]:
        return [pathlib.Path(f) for f in glob.glob(value)]


def _get_metafile(raw_data_dir: pathlib.Path) -> pathlib.Path:
    granule_dir = raw_data_dir.joinpath("GRANULE").glob("*")
    granule_dir = next(granule_dir)
    meta_file = granule_dir.joinpath("MTD_TL.xml")
    return meta_file


def _get_node(root, node_name):
    for node in root:
        if node_name in node.tag:
            return node
    return None


def _get_mean_angles(metafile: pathlib.Path) -> Tuple[float, float]:
    tree = ET.parse(metafile)
    root = tree.getroot()
    geometric_info = _get_node(root, "Geometric_Info")
    tile_angles = _get_node(geometric_info, "Tile_Angles")
    mean_angles = _get_node(tile_angles, "Mean_Sun_Angle")
    for angle in mean_angles:
        if "ZENITH" in angle.tag:
            zenith = float(angle.text)
        elif "AZIMUTH" in angle.tag:
            azimuth = float(angle.text)
        else:
            raise ValueError("Unkown angle: {}".format(angle.tag))
    return zenith, azimuth


def _get_grid_dimensions(metafile: pathlib.Path) -> Generator[int, int, int]:
    tree = ET.parse(metafile)
    root = tree.getroot()
    geometric_info = _get_node(root, "Geometric_Info")
    geocoding = _get_node(geometric_info, "Tile_Geocoding")
    for node in geocoding:
        if node.tag == "Size":
            nrows = ncols = None
            for el in node:
                if el.tag == "NROWS":
                    nrows = int(el.text)
                elif el.tag == "NCOLS":
                    ncols = int(el.text)
                else:
                    raise ValueError("Unknown element '{}' in node '{}'".format(el.tag, node))
            resolution = int(node.get("resolution"))
            yield (nrows, ncols, resolution)


def _get_resolution(shape, meta_file):
    resolution = None
    erows, ecols = shape
    for nrows, ncols, res in _get_grid_dimensions(meta_file):
        if nrows == erows and ncols == ecols:
            resolution = res
    if resolution is None:
        raise ValueError(
                "Could not find a resolution for elevation model of shape {}".format(shape))
    return resolution


def xyray_vector(azimuth, transform):
    azimuth = np.deg2rad(azimuth)
    x, y = np.sin(azimuth), np.cos(azimuth)
    x, y = transform * (x, y)
    x -= transform.xoff
    y -= transform.yoff
    m = y/x
    if m < 1. and m > -1.:
        y = m
        x = 1.
    else:
        y = 1.
        x = 1./m
    return x, y


def _run_shader(raw_data_dir, elevation_model, transform, resolution, chunk_size, num_workers):
    meta_file = _get_metafile(raw_data_dir)
    zenith, azimuth = _get_mean_angles(meta_file)
    ray = xyray_vector(azimuth, transform)

    def worker(ystart):
        yend = min(ystart+chunk_size, elevation_model.shape[0])
        shadow = hillshade(elevation_model, zenith, ray, resolution, ystart, yend)
        return shadow

    with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
        ystart = np.arange(0, elevation_model.shape[0], chunk_size)
        results = executor.map(worker, ystart)
        shadow = []
        pbar = tqdm.tqdm(total=ystart.shape[0], desc="Image chunks")
        for res in results:
            shadow.append(res)
            pbar.update()
        pbar.close()
        shadow = np.vstack(shadow)
    return shadow


@click.command()
@click.argument('elevation_infile', type=click.Path(file_okay=True))
@click.argument('s2_dirs', type=GlobbityGlob())
@click.argument('shaded_outfile', type=click.Path(file_okay=True))
@click.option("--chunk-size", default=100, type=int,
              help="""Chunk size of the image in y direction that is processed at a time.\
 Only affects the progress bar update frequency. Defaults to 100.""")
@click.option("--workers", default=4, help="Number of workers. Defaults to 4.")
def cli(elevation_infile, s2_dirs, shaded_outfile, chunk_size, workers):
    """Calculates shaded regions based on and elevation model and incident angles
    that are read from S2 raw data directories.

    Parameters:

        elevation_infile:   elevation model (.tif)

        s2_dirs:            S2 raw data directories

        shaded_outfile:     output file (.tif)
    """

    with rasterio.open(elevation_infile, 'r') as src:
        profile = src.profile.copy()
        elevation_model = src.read()[0]
        transform = src.transform
        resolution = src.res

    shadow = np.zeros(elevation_model.shape, dtype=int)
    for raw_data_dir in tqdm.tqdm(s2_dirs, desc="Angles"):
        shadow += _run_shader(
                raw_data_dir, elevation_model, transform, resolution, chunk_size, workers)
    shadow = shadow.astype(np.float32)
    shadow /= shadow.max()

    profile.update(
        dtype=shadow.dtype,
        count=1,
        compress='lzw',
        nodata=None)
    with rasterio.open(shaded_outfile, 'w', **profile) as dst:
        dst.write(shadow, 1)
