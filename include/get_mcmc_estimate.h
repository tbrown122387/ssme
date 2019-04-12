#ifndef GET_MCMC_ESTIMATE_H
#define GET_MCMC_ESTIMATE_H

#include <Eigen/Dense>
#include <string>
#include <iostream> // for cerr
#include <fstream>

#include "param_pack.h"


// TODO handle if certain rows have fewer columns (this was an error)
//! make a point estimate from samples
/**
 * @brief takes the average of your mcmc samps to come up with an average
 * @param file_loc the path to where the samples are stored (in a certain format)
 * @param burn_in how many rows that are discarded before taking the average
 * @param tts a vector of transformation types, in case the *transformed* samples are stored
 * @param from_trans are the samples stored in a transformed way
 * @param to_trans do you want an estimate of the nontransformed, or transformed?
 */
Eigen::VectorXd ave_mcmc_samps(const std::string& file_loc, int burn_in, const std::vector<TransType>& tts, bool from_trans, bool to_trans);


#endif // GET_MCMC_ESTIMATE_H
