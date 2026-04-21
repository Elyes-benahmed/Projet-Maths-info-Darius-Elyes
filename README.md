# Mini-projet

This module contains the python modules to be completed for the project.

## Project organization

```
./
├── imgs/ # Contains images required for the project
├── src/ # Source code of the project
├── script.py # Script for the signal processing part
├── script_optim.py # Script for the optimization part -- to be completed
└── README.md # Project documentation
```

## Overview

The `wavelet.py` module in src/ provides a `wavelet` class that allows you to 
create and access the filter coefficients of any wavelet supported by 
the library PyWavelets.

The class automatically extracts the **decomposition** and **reconstruction** 
filters for the given wavelet, including:

- `dec_low`: decomposition low-pass filter
- `dec_high`: decomposition high-pass filter
- `rec_low`: reconstruction low-pass filter
- `rec_high`: reconstruction high-pass filter

Note: you don't need to modify `wavelet.py` for the project.

The `dwt.py` module in src/ serves as a template for the implementation of the
wavelet decomposition.

The `blind_deconv.py` module in src/ serves as an input file with functions to be used
for the optimization part of the project.


## Requirements

Install dependencies via pip:

```bash
pip install numpy pywavelets
```



