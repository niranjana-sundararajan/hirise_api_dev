# HIRISEImgs Package
HIRISE Image Data, Information, Preprocessing and Models\
![image](https://user-images.githubusercontent.com/88569855/187724785-28a58661-9707-40d1-a130-975b57708e2b.png)

\
This is a python tool that is used to to query,filter,pre-process and download HIRISE
images from NASA's Planetray Data System(PDS) and University of Arizona's HiWish Databse
and extract hidden patterns and features in the images over 10 preprocessing methods including,
tiling, remove blank space, grayscalimg, dynamic resizing, and multiple dimesnion reduction
methods including principal component analysis, UMAP and t-SNE.
The images can be encoded using convolutional autoencoders, InceptionV3 and Xception networks and
finally the using can call 36 different combination models to cluster the pre-process images as
desired.

# Package Structure
<img width="154" alt="file_structure" src="https://user-images.githubusercontent.com/88569855/187727446-eb610e3b-f103-4228-a7fb-b1120e67ad7e.png">

## Installation Instructions

For Mac, Windows or Linux users:

```
pip install HIRISEimgs
```

To use on colab :

```
!pip install HIRISEimgs
```

For developer who wish to download and use the code locally, download the HIRISE_api folder\
or use the command `cd HIRISE_api` to enter into the codebase.\
\
Next, following the steps listed below

1. Upgrade your pip environment
```
pip install --upgrade pip
```
2. Install Requirements
```
pip install -r requirements.txt
```
3. Install all modules
```
pip install -e .
```

## Usage
Please refer the [example workflow](examples/Example_Workflow_Hirise_Package.ipynb) in the examples folder.
The [doumentation](docs/pdf) can further help you understand the current cope of the package functionalities.


## Support
You can [open an issue](https://github.com/ese-msc-2021/irp-ns1321/issues) for support.

## Contributing
Contribute using [Github Flow](https://guides.github.com/introduction/flow/). 
Create a branch, add commits, and [open a pull request](https://github.com/ese-msc-2021/irp-ns1321/compare).

## Documentation
The documentation is dynamically updated using Sphinx and can be accessed in the [pdf version](docs/pdf) 
or the [html version](docs/html) in the docs folder.

## License
MIT License, for more details click [here](HIRISE_api/LICENSE.txt)

## Version
Initial Release 1.0.0
