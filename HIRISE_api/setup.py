from setuptools import setup, find_packages

PCKGNAME = "HIRISEimgs"
VERSION = '0.0.0.0.dev23'
DESCRIPTION = 'HIRISE Image data and information'

# Setting up
setup(
    name=PCKGNAME,
    version=VERSION,
    author="Niranjana Sundararajan",
    author_email="<niranjanasundararajan@gmail.com>",
    description=DESCRIPTION,
    license='MIT',
    packages=find_packages(),
    package_dir= {'hirise' : "./hirise"},
    include_package_data=True ,
    install_requires=['humanize', 'requests', 'bs4','pyamg'],
    keywords=['python', 'HIRISE', 'NASA', 'PDS', 'API', 'image data'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)