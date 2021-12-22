from os import environ
from os.path import join, abspath, dirname
from setuptools import setup, find_packages

root_path = abspath(dirname(__file__))

assert 'BACKEND' in environ
assert 'GPU' in environ

with open(join(root_path, 'docker', 'gpu' if int(environ['GPU']) == 1 else 'cpu' , environ['BACKEND'], 'requirements.txt')) as f:
    REQUIREMENTS = [line.strip() for line in f]

setup(name="matchvec", 
    version=environ['CI_COMMIT_TAG'],
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    )
