import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="sinnn",
    version="0.0.1",
    description="Numpy powered neural network library",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/sohaibimran7/sinnn",
    author="Sohaib Imran",
    author_email="sohaibimran7@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy"],
    python_requires='>=3.6',
)
