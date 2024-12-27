from setuptools import setup, find_packages

__version__ = "1.1.6"

setup(
    name='sgptools',
    version=__version__,
    packages=find_packages(),
    python_requires='>=3.6',
    url='https://www.itskalvik.com/sgp-tools',
    license='Apache-2.0',
    author_email='itskalvik@gmail.com',
    author='Kalvik',
    description='Software Suite for Sensor Placement and Informative Path Planning',
    long_description='Software Suite for Sensor Placement and Informative Path Planning',
    install_requires=['apricot-select',
                      'matplotlib',
                      'pandas',
                      'scikit-learn',
                      'scipy',
                      'numpy<2.0.0',
                      'ortools',
                      'scikit-image',
                      'shapely',
                      'cma',
                      'bayesian-optimization',
                      'hkb_diamondsquare',
                      'tensorflow-probability[tf]>=0.21.0',
                      'tensorflow>=2.13.0; platform_machine!="arm64"',
                      'tensorflow-aarch64>=2.13.0; platform_machine=="arm64"',
                      'typing_extensions',
                      'gpflow>=2.7.0',
                      'pillow'
                     ]
)
