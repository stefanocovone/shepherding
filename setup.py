from setuptools import setup

setup(
    name="shepherding",
    version="0.0.2",
    packages=["shepherding"],
    install_requires=["gymnasium>=0.26.1", "pygame>=2.1.0, <3.0.0", "numpy"]
)