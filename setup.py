from setuptools import setup

setup(
    name='hillshade',
    version='0.1',
    py_modules=['hillshade'],
    install_requires=[
        'Click',
        'Numpy',
        'tqdm',
        'rasterio',
    ],
    entry_points='''
        [console_scripts]
        hillshade=hillshade.cli:cli
    ''',
)