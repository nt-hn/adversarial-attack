# adversarial-attack

adversarial-attack is a project for adding a patch on a face and making the face unrecognizable by machines

## Installation

Use the package manager pip to install the following modules 

```bash

pip install numpy python-math torch torchvision models==0.9.3 opencv-python imutils dlib

```

### Incase you run into problems while trying to install dlib 

>First use the package manager pip to install the following module

```bash

pip install cmake

```

> Make sure you have visual studio c++ for cmake installed

> Then use the package manager pip to install dlib

```bash

pip install dlib

```

## Usage

Make sure to change the picture files of the person and the patch. In addition make sure that the image of the person is 128px*128px.
The program returns the patch on the face of the desired person making the person's picture unrecognizable by machines.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
