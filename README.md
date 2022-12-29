# Fast-BEM

This library provides a fast version of the Boundary Element Method (BEM) for the simulation of bubble resonances, specifically the low-frequency resonances of a rectangular array of air bubbles in water.

This library accompanies the publication "Modeling frequency shifts of collective bubble resonances with the boundary element method." A preprint is available on arXiv:2209.01245 (https://arxiv.org/abs/2209.01245).

## Installation

The library uses the BEM implementation of the bempp-cl package (https://github.com/bempp/bempp-cl). Please follow the installation instructions on the official website (https://bempp.com/installation.html). The code was tested with version 0.2.4 of bempp-cl. In addition, standard Python libraries such as pandas, seaborn and pickle are used for postprocessing and visualisation.

## Instructions

The main purpose of the library is analyzing the low-frequency resonances of a rectangular array of air bubbles in water. Data for the configurations used in the publication mentioned above have been precomputed and stored. The frequency response of these configurations can directly be plotted with the Jupyter notebook "frequency sweeps/plotter.ipynb". Other configurations can be simulated with the Jupyter notebook "frequency sweeps/generator.ipynb" specifying the number of bubbles on each side of the rectangular array and the separation distance. Multiple runs of frequency sweeps can be performed and the results are added to a data base in the folder "frequency sweeps/pickles", which can be visualised with the plotter notebook.

The acoustic field near the bubble array can be visualized with the Jupyter notebook "3DGraphs/generator.ipynb".

The far field response of the bubble array can be visualized with the polar plots available in the Jupyter notebook "polar plots/generator.ipynb".

The folder "packages" provides the algorithm itself, including the BEM formulation and the custom acceleration for rectangular arrays.

Simulating arrays with few bubbles and a coarse mesh can be performed on a standard laptop computer. Analyzing large arrays with a fine mesh at many frequencies requires a high-performance desktop machine.

## More information

Please read the publication "Modeling frequency shifts of collective bubble resonances with the boundary element method" for more information, see arXiv:2209.01245 (https://arxiv.org/abs/2209.01245). For queries, please feel free to use GitHub's functionality or contact the authors directly.
