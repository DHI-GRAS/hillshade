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
    """Finds the GRANULE meta file in a S2 data directory."""
    granule_dir = raw_data_dir.joinpath('GRANULE')
    granule_sub_dir = granule_dir.glob('*')
    try:
        granule_sub_dir = next(granule_sub_dir)
    except StopIteration:
        raise IOError(f'subdirectory of "{granule_dir}" does not exist.')
    meta_file = granule_sub_dir.joinpath('MTD_TL.xml')
    if not meta_file.exists():
        raise IOError(f'GRANULE meta file "{meta_file}" does not exist.')
    return meta_file


def _get_mean_angles(metafile: pathlib.Path) -> Tuple[float, float]:
    """Retrieves zenith and azimuth (in degrees) from a GRANULE metafile."""
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


def rasterize(azimuth, transform=None):
    """Convert the azimuth into its components on the XY-plane. Depending on the value of the
    azimuth either the x or the y component of the resulting vector is scaled to 1, so that
    it can be used conveniently to walk a grid.
    """
    azimuth = np.deg2rad(azimuth)
    xdir, ydir = np.sin(azimuth), np.cos(azimuth)

    if transform is not None:
        xdir, ydir = transform * (xdir, ydir)
        xdir -= transform.xoff
        ydir -= transform.yoff

    slope = ydir / xdir
    if slope < 1. and slope > -1.:
        xdir = 1.
        ydir = slope
    else:
        xdir = 1. / slope
        ydir = 1.
    return xdir, ydir


def run_shader(elevation_model, transform, resolution, zenith, azimuth, chunk_size=100, workers=4):
    """Calculate shaded regions based on the elevation model and the incident angles of the sun.

    Params:
        elevation_model (np.ndarray):
            Two-dimensional array specifying the elevation at each point of the grid.
        transform (rasterio.affine.Affine):
            Affine transform of the elevation_model
        resolution (tuple):
            resolution in meters of the elevation_model
        zenith (float):
            zenith in degrees
        azimuth (float):
            azimuth in degrees
        chunk_size (int):
            splits the shading calculation into chunk sized tiles in the y-direction
        workers (int):
            number of threads to spawn
    """
    ray = rasterize(azimuth, transform)

    def worker(ystart):
        yend = min(ystart + chunk_size, elevation_model.shape[0])
        shadow = hillshade(elevation_model, resolution, zenith, ray, ystart, yend)
        return shadow

    with concurrent.futures.ThreadPoolExecutor(workers) as executor:
        ystart = np.arange(0, elevation_model.shape[0], chunk_size)
        results = executor.map(worker, ystart)
        shadow = []
        with tqdm.tqdm(total=ystart.shape[0], desc='Image chunks') as pbar:
            for res in results:
                shadow.append(res)
                pbar.update()
        shadow = np.vstack(shadow)
    return shadow


@click.command()
@click.argument('s2_dirs', type=GlobbityGlob())
@click.option(
    '--elevation-infile', "-i", type=click.Path(dir_okay=False), required=True,
    help='Input elevation model (.tif)')
@click.option(
    '--shaded-outfile', "-o", type=click.Path(dir_okay=False), required=True,
    help='Output file with shaded regions. The output is one where there is always shade'
    'and zero where the sun always shines!')
@click.option(
    '--chunk-size', default=100, type=int, show_default=True,
    help='Chunk size of the image in y direction that is processed at a time.'
    'Only affects the progress bar update frequency.')
@click.option('--workers', default=4, show_default=True, help='Number of workers.')
def cli(s2_dirs, elevation_infile, shaded_outfile, chunk_size, workers):
    """Calculates shaded regions based on and elevation model and incident angles
    that are read from S2 raw data directories."""
    if not s2_dirs:
        raise IOError('S2 directory does not exist.')

    with rasterio.open(elevation_infile, 'r') as src:
        profile = src.profile.copy()
        elevation_model = src.read(1)
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
