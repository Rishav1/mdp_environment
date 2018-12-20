#!/usr/bin/env python

from distutils.core import setup

setup(name='mdp_environment',
      version='1.0',
      description='Environment for Markov Decision Processes(primarily for RL)',
      author='Rishav Chourasia',
      author_email='rishav.chourasia@gmail.com',
      license='MIT',
      url='https://github.com/Rishav1/mdp_environment',
      py_modules=['mdp_environment'],
      install_requires=['numpy', 'pytest', 'networkx', 'pydot', 'GraphViz', 'gym', 'scipy', 'matplotlib'],
      zip_safe=False)
