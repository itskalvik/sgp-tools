from setuptools import setup, find_packages

__version__ = "2.0.0"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

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
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ]        
)
