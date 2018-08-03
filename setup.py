from setuptools import setup

setup(
    name='hillshade',
    version='0.1',
    py_modules=['hillshade'],
    install_requires=[
        'lxml',
        'Click',
        'Numpy',
        'Numba',
        'tqdm',
        'rasterio',
    ],
    extras_require={
        'test': [
            'pytest>=3.5',
            'pytest-cov',
            'pytest-mypy',
            'pytest-flake8',
            'codecov',
            'attrs>=17.4.0',
        ],
    },
    entry_points='''
        [console_scripts]
        hillshade=hillshade.cli:cli
    ''',
)
