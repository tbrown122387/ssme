# SSME: a c++ static library for the estimation of state space models

The current availability includes:

1. particle marginal Metropolis-Hastings with generic parameter proposal distributions.

2. particle marginal Metropolis-Hastings with multivariate normal random-walk proposals. 

Also included is the `paramPack` container class that holds onto parameters. Because it is quite common to transform the parameters, this class has the ability to return transformed and untransformed parameters vectors, as well as automatically compute and return (log- )jacobians (which comes in handy when working with evaluating the parameter proposal distribution).
