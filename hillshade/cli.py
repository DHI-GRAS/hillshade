import os
import glob
import pathlib
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


@click.command()
@click.argument('elevation_infile', type=click.Path(file_okay=True))
@click.argument('s2_dirs', type=GlobbityGlob())
@click.argument('shaded_outfile', type=click.Path(file_okay=True))
@click.option("--chunk-size", default=None, type=int,
              help="""Chunk size of the image in y direction that is processed at a time.\
 Only affects the progress bar update frequency. Defaults to 100.""")
def cli(elevation_infile, s2_dirs, shaded_outfile, chunk_size=None):
    """Calculates shaded regions based on and elevation model and incident angles
    that are read from S2 raw data directories.

    Parameters:

        elevation_infile:   elevation model (.tif)

        s2_dirs:            S2 raw data directories

        shaded_outfile:     output file (.tif)
    """
    if chunk_size is None:
        chunk_size = 100

    with rasterio.open(elevation_infile, 'r') as src:
        profile = src.profile.copy()
        elevation_model = src.read()[0]

    shades = []
    shadow = np.zeros(elevation_model.shape, dtype=np.int32)
    for raw_data_dir in tqdm.tqdm(s2_dirs):

        meta_file = _get_metafile(raw_data_dir)
        zenith, azimuth = _get_mean_angles(meta_file)

        resolution = None
        erows, ecols = elevation_model.shape
        for nrows, ncols, res in _get_grid_dimensions(meta_file):
            if nrows == erows and ncols == ecols:
                resolution = res
        if resolution is None:
            raise ValueError(
                    "Could not find a resolution for elevation model of shape {}"
                    .format(elevation_model.shape))

        shadow[:,:] = 0
        for ystart in tqdm.tqdm(np.arange(0, elevation_model.shape[0], chunk_size)):
            yend = min(ystart+chunk_size, elevation_model.shape[0])
            hillshade(elevation_model, shadow, zenith, azimuth, resolution, ystart, yend)
        shades.append(shadow.copy())

    shadow = np.sum(shades, axis=0)
    shadow = shadow.astype(float)
    shadow /= shadow.max()

    profile.update(
        dtype=np.float64,
        count=1,
        compress='lzw',
        nodata=None)
    with rasterio.open(shaded_outfile, 'w', **profile) as dst:
        dst.write(shadow, 1)
