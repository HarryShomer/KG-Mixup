import os
from setuptools import setup


def read():
    return open(os.path.join(os.path.dirname(__file__), 'README.md')).read()

setup(
    name='kgpy',
    version='0.0.1',
    url='https://github.com/HarryShomer/kgpy',
    author='Harry Shomer',
    author_email='Harryshomer@gmail.com',
    packages=['kgpy'],
    install_requires=['tqdm', 'tensorboard', 'torch', 'torch_geometric', 'torch_scatter', 'optuna'],
    include_package_data=True,
    zip_safe=False
)