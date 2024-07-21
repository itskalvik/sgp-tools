from setuptools import setup, find_packages

__version__ = "1.0.2"

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
    install_requires=['apricot-select==0.6.0',
                      'matplotlib==3.5.2',
                      'pandas==1.4.3',
                      'scikit-learn==1.1.1',
                      'scipy==1.8.1',
                      'tensorflow==2.13.0',
                      'tensorflow_probability==0.21.0',
                      'gpflow==2.7.0',
                      'numpy==1.23.5',
                      'ortools==9.2.9972',
                      'scikit-image',
                      'shapely',
                      'cma',
                      'bayesian-optimization',
                      'pillow',
                      'hkb_diamondsquare']
)
