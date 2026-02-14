from setuptools import setup, find_packages

__version__ = "2.0.7"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='sgptools',
    version=__version__,
    packages=find_packages(),
    python_requires='>=3.6',
    url='https://www.SGP-Tools.com',
    license='Apache-2.0',
    author_email='itskalvik@gmail.com',
    author='Kalvik',
    description='A Python library for efficient sensor placement and informative path planning',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['apricot-select',
                      'matplotlib',
                      'pandas',
                      'scikit-learn',
                      'scipy',
                      'numpy<2.0.0',
                      'numba',
                      'ortools<9.14',
                      'scikit-image',
                      'shapely',
                      'cma',
                      'bayesian-optimization',
                      'hkb_diamondsquare',
                      'tensorflow-probability[tf]>=0.25.0',
                      'tensorflow>=2.18.0,<2.20.0; platform_machine!="arm64"',
                      'tensorflow-aarch64>=2.18.0,<2.20.0; platform_machine=="arm64"',
                      'tensorflow-macos>=2.18.0,<2.20.0; platform_system=="Darwin" and platform_machine=="arm64"',
                      'typing_extensions',
                      'gpflow>=2.10.0',
                      'pillow',
                      'geopandas',
                      'setuptools<81'
                     ],
        classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ]        
)
