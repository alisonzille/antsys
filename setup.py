from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    readme = fh.read()

setup(
  name='antsys',
  version='0.1.44',
  url='https://github.com/alisonzille/antsys',
  description='AntSys - General Purpose Ant Colony Optimization System',
  author='Alison Zille Lopes',
  author_email='alisonzille@gmail.com',
  license='GNU General Public License v3 (GPLv3)',
  long_description=readme,
  long_description_content_type='text/markdown',
  keywords=['ACO', 'optimization', 'ant'],
  packages=find_packages(),
  install_requires=['numpy'],
)
