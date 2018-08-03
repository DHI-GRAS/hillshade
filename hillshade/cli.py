import glob
import pathlib
import concurrent.futures
from typing import Any, List, Tuple

from lxml import etree
import numpy as np
import rasterio
import tqdm
import click
from hillshade.hillshade import hillshade


class GlobbityGlob(click.ParamType):
    """Expands a glob pattern to Path objects"""
    name = 'glob'

    def convert(self, value: str, *args: Any) -> List[pathlib.Path]:
        return [pathlib.Path(f) for f in glob.glob(value)]


def _get_metafile(raw_data_dir: pathlib.Path) -> pathlib.Path:
    granule_dir = raw_data_dir.joinpath('GRANULE')
    granule_sub_dir = granule_dir.glob('*')
    try:
        granule_sub_dir = next(granule_sub_dir)
    except StopIteration:
        raise IOError(f'GRANULE subdirectory "{granule_dir}\*" does not exist.')
    meta_file = granule_sub_dir.joinpath('MTD_TL.xml')
    if not meta_file.exists():
        raise IOError(f'GRANULE meta file "{meta_file}" does not exist.')
    return meta_file


def _get_mean_angles(metafile: pathlib.Path) -> Tuple[float, float]:
    tree = etree.parse(str(metafile))
    root = tree.getroot()

    mean_angle_tag = 'n1:Geometric_Info/Tile_Angles/Mean_Sun_Angle'
    mean_angles = root.find(mean_angle_tag, root.nsmap)
    if mean_angles is None:
        raise ValueError(f'Could not find "{mean_angle_tag}" in meta file.')

    zenith = mean_angles.find("ZENITH_ANGLE")
    azimuth = mean_angles.find("AZIMUTH_ANGLE")
    if zenith is None or azimuth is None:
        raise ValueError(f'Could not find angles in {mean_angles}')

    zenith = float(zenith.text)
    azimuth = float(azimuth.text)
    return zenith, azimuth


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


def run_shader(elevation_model, transform, resolution, zenith, azimuth, chunk_size=100, workers=4):
    ray = xyray_vector(azimuth, transform)

    def worker(ystart):
        yend = min(ystart+chunk_size, elevation_model.shape[0])
        shadow = hillshade(elevation_model, zenith, ray, resolution, ystart, yend)
        return shadow

    with concurrent.futures.ThreadPoolExecutor(workers) as executor:
        ystart = np.arange(0, elevation_model.shape[0], chunk_size)
        results = executor.map(worker, ystart)
        shadow = []
        pbar = tqdm.tqdm(total=ystart.shape[0], desc='Image chunks')
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
@click.option('--chunk-size', default=100, type=int,
              help='''Chunk size of the image in y direction that is processed at a time.\
 Only affects the progress bar update frequency. Defaults to 100.''')
@click.option('--workers', default=4, help='Number of workers. Defaults to 4.')
def cli(elevation_infile, s2_dirs, shaded_outfile, chunk_size, workers):
    """Calculates shaded regions based on and elevation model and incident angles
    that are read from S2 raw data directories.

    Parameters:

        elevation_infile:   elevation model (.tif)

        s2_dirs:            S2 raw data directories

        shaded_outfile:     output file (.tif)
    """
    if not s2_dirs:
        raise IOError('S2 directory does not exist.')

    with rasterio.open(elevation_infile, 'r') as src:
        profile = src.profile.copy()
        elevation_model = src.read()[0]
        transform = src.transform
        resolution = src.res

    shadow = np.zeros(elevation_model.shape, dtype=int)
    for raw_data_dir in tqdm.tqdm(s2_dirs, desc='Angles'):
        meta_file = _get_metafile(raw_data_dir)
        zenith, azimuth = _get_mean_angles(meta_file)
        shadow += run_shader(
                elevation_model, transform, resolution, zenith, azimuth, chunk_size, workers)
    shadow = shadow.astype(np.float32)
    shadow /= shadow.max()

    profile.update(
        dtype=shadow.dtype,
        count=1,
        compress='lzw',
        nodata=None)
    with rasterio.open(shaded_outfile, 'w', **profile) as dst:
        dst.write(shadow, 1)
