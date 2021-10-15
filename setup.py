from os import environ
from os.path import join, abspath, dirname
from setuptools import setup, find_packages

root_path = abspath(dirname(__file__))

with open(join(root_path, 'requirements.txt')) as f:
    REQUIREMENTS = [line.strip() for line in f]

setup(name="matchvec", 
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    entry_points={'console_scripts': [
          'matchvecapp = matchvecapp.app:main'
      ]},)
