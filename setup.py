from os import environ
from os.path import join, abspath, dirname
from setuptools import setup, find_packages

root_path = abspath(dirname(__file__))

assert 'BACKEND' in environ
assert 'GPU' in environ

with open(join(root_path, 'gpu' if int(environ['GPU']) == 1 else 'cpu' , environ['BACKEND'], 'requirements.txt')) as f:
    REQUIREMENTS = [line.strip() for line in f]

setup(name="matchvec", 
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    entry_points={'console_scripts': [
          'matchvecapp = matchvecapp.app:main'
      ]},)
