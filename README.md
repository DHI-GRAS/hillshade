# hillshade

Calculates shaded regions based on an elevation model and the zenith/azimuth of the sun.
The scripts needs and elevation model and one or more S2 data directories to read the angles from.
Usage:

    hillshade -i elevation_model.tif -o hillshade.tif s2_data/*.SAFE

#### Note: The elevation model must be in UTM projection!

Install the package with:

    git clone https://github.com/DHI-GRAS/hillshade
    cd hillshade
    pip install .

The hillshader is parallelized over row-chunks of the elevation model.
Chunk size and workers can be adjusted by passing `--chunk-size` and `--worker` flags.
