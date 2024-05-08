from setuptools import setup, find_packages

__version__ = "1.0.0"

setup(
    name='sgptools',
    version=__version__,
    packages=find_packages(),
    python_requires='>=3.6',
    url='https://www.itskalvik.com/sgp-tools',
    license='Apache-2.0',
    author_email='itskalvik@gmail.com',
    author='Kalvik Jakkala',
    description='Software suite for Sensor Placement and Informative Path Planning',
)