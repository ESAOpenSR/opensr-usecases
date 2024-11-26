from setuptools import setup, find_packages

setup(
    name='opensr-usecases',
    version='0.0.10',
    packages=find_packages(),
    install_requires=[
        # List of dependencies
        'numpy', 
        'torch',
        "tqdm",
        "prettytable",
        "Pillow",
    ],
    author='Simon Donike',
    author_email='simon@donike.net',
    description='Use-case-based validation for SR products.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ESAOpenSR/opensr-usecases',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)