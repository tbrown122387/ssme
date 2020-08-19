# SSME: a c++ static library for the estimation of state space models

[![DOI](https://zenodo.org/badge/154373836.svg)](https://zenodo.org/badge/latestdoi/154373836)

The current availability includes particle marginal Metropolis-Hastings with a multivariate normal parameter proposals. Also included is the `pack` container class template that holds onto parameters. Because it is quite common to transform the parameters, this class has the ability to return transformed and untransformed parameters vectors, as well as automatically compute and return (log- )jacobians (which comes in handy when working with evaluating the parameter proposal distribution).


## Installation

`cd` into the location where you want the code saved and then type the following

    git clone https://github.com/tbrown122387/ssme.git
    cd ssme
    mkdir build && cd build/
    cmake .. -DCMAKE_INSTALL_PREFIX:PATH=/usr/local
    sudo cmake --build . --config Release --target install --parallel




## example:

For an example, navigate to the `ssme/example` directory, and type the following

```
mkdir build && cd build
cmake ..
make
./main spy_returns.csv univ_svol_pmmh_samples univ_svol_pmmh_messages 1000
```
