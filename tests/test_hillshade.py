import os
import click
from click.testing import CliRunner
import numpy as np
import affine
import rasterio
from hillshade import cli


def test_cli():
    runner = CliRunner()
    dirname = os.path.dirname(os.path.realpath(__file__))
    s2dir = os.path.join(dirname, "s2dir")

    infile = "elevation_model.tif"
    outfile = "shadow.tif"

    model_shape = (10, 10)
    data = np.zeros(model_shape, dtype=np.float32)
    data[5, :] = 2

    true_shadow = np.zeros(model_shape, np.float32)
    true_shadow[4, 1:] = 1.

    with runner.isolated_filesystem():

        profile = dict(
            transform=affine.Affine(2, 0, 0, 0, -2, 10000),
            crs={'init': 'epsg:3857'},
            count=1,
            height=data.shape[0],
            width=data.shape[1],
            nodata=0,
            dtype='float32',
            driver='GTiff')

        with rasterio.open(infile, 'w', **profile) as dst:
            dst.write(data, 1)

        result = runner.invoke(cli.cli, [s2dir, "-i", infile, "-o", outfile])

        with rasterio.open(outfile, 'r') as src:
            shadow = src.read(1)

        assert result.exit_code == 0
        assert np.all(shadow == true_shadow)
