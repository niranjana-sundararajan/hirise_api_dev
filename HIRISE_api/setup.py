from setuptools import setup, find_packages

PCKGNAME = "HIRISEimgs"
VERSION = '0.0.0.0.dev75'
DESCRIPTION = 'HIRISE Image data and information'

# Setting up
setup(
    name=PCKGNAME,
    version=VERSION,
    python_requires='>3.5.2',
    author="Niranjana Sundararajan",
    author_email="<niranjanasundararajan@gmail.com>",
    description=DESCRIPTION,
    license='MIT',
    packages=find_packages(),
    package_dir= {'hirise' : "hirise"},
    include_package_data=True ,
    install_requires=['wheel','humanize', 'requests', 'beautifulsoup4','umap-learn', 'hdbscan','rasterfairy'],
    keywords=['python', 'HIRISE', 'NASA', 'PDS', 'API', 'image data'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
