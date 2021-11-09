# V-System: Vascular Lindenmayer Systems

V-System if a Python project to generate synthetic vascular networks which utilises [Linenmayer Systems](https://en.wikipedia.org/wiki/L-system), known as L-Systems; a type of [formal grammar](https://en.wikipedia.org/wiki/Formal_grammar).

V-System uses the grammars of [M. Galarreta-Valverde et al. (2013)](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/8669/86691I/Three-dimensional-synthetic-blood-vessel-generation-using-stochastic-L-systems/10.1117/12.2007532.full?SSO=1) who extended the traditional L-System by adding stochastic rules and parameters in the grammar to generate synthetic 3D blood vessels.

The V-System is split into three processes:
* Generate a grammatical string for a user-defined number of iterations.
* Convert the string into a list of coordinates.
* Perform 3D voxel traversal to generate a 3D binary mask of the generated vascular network.

An example of generated synthetic vascular trees of increasing complexity is shown below.

![alt text](https://github.com/psweens/V-System/blob/master/Lnet_Generations.jpg)

## Installation
V-System is compatible with C++11, and has been tested on Ubuntu 18.04 LTS and macOS Big Sur. 
Other distributions of Linux and Windows should work as well.

To install V-System from source, download zip file on GitHub page or run the following in a terminal:
```bash
git clone https://github.com/psweens/V-System.git
```

The required Python packages can be found [here](https://github.com/psweens/V-System/blob/master/REQUIREMENTS.txt). The package list can be installed, for example, using creating a Conda environment by running:
```bash
conda create --name <env> --file REQUIREMENTS.txt
```

## Acknowledgements
I would like to acknowledge that the V-system code utilises several portions of code originally written by Miguel A. Galarreta-Valverde [here](https://teses.usp.br/teses/disponiveis/45/45134/tde-30112012-172822/pt-br.php).
