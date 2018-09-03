from setuptools import setup, find_packages
import versioneer

setup(
    name='hillshade',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
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
        hillshade=hillshade.scripts.cli:cli
    ''',
)
