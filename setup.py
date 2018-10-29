#!/usr/bin/env python

from setuptools import setup

setup(name='adjustTextOpen',
      version='0.7.4b',
      description='Iteratively adjust text position in matplotlib plots to minimize overlaps',
      author='Ilya Flyamer, John C. Thomas',
      author_email='flyamer@gmail.com, jcthomas000@gmail.com',
      url='https://github.com/johncthomas/adjustTextOpen',
      packages=['adjustTextOpen'],
      install_requires=['numpy']
     )
