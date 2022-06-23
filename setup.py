# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name="pygrad",
    version="0.0.1",
    description="A minimal, fast, scalar-valued, dependency-free Python library for automatic "
                "differentiation.",
    long_description=readme,
    author="Kai Fabi",
    author_email="kai.fabi@posteo.net",
    url='https://github.com/KaiFabi/PyAutograd',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)