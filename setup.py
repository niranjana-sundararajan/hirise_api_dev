from setuptools import setup, find_packages

PCKGNAME = "HIRISEimgs"
DESCRIPTION = 'HIRISE Image Data, Information, Preprocessing and Models'
LONG_DESCRIPTION = 'Package created to support thesis : "Deep Learning \
    Algorithms Applied to Surface Mapping of Mars" written in part for \
        completion of Masters degree at Imperial College London'
# Setting up
setup(
    name=PCKGNAME,
    version="1.0.0",
    python_requires='>3.5.2',
    author="Niranjana Sundararajan",
    author_email="<niranjanasundararajan@gmail.com>",
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    license='MIT',
    package_dir={
        '': "HIRISE_api"},
    packages=find_packages(
        where="HIRISE_api"),
    include_package_data=True,
    install_requires=[
        'wheel',
        'humanize',
        'requests',
        'beautifulsoup4',
        'umap-learn',
        'hdbscan',
        'rasterfairy',
        'pytest'],
    keywords=[
        'python',
        'HIRISE',
        'NASA',
        'PDS',
        'API',
        'image data'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ])
