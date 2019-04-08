# SSME: a c++ static library for the estimation of state space models

[![DOI](https://zenodo.org/badge/154373836.svg)](https://zenodo.org/badge/latestdoi/154373836)

The current availability includes:

1. particle marginal Metropolis-Hastings with generic parameter proposal distributions.

2. particle marginal Metropolis-Hastings with multivariate normal random-walk proposals. 

Also included is the `paramPack` container class that holds onto parameters. Because it is quite common to transform the parameters, this class has the ability to return transformed and untransformed parameters vectors, as well as automatically compute and return (log- )jacobians (which comes in handy when working with evaluating the parameter proposal distribution).

## example:

For an example, navigate to the `ssme/example` directory, type `make`, and then something like 

```
./main spy_returns.csv univ_svol_pmmh_samples univ_svol_pmmh_messages 1000
```
