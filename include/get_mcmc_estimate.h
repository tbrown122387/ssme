#ifndef GET_MCMC_ESTIMATE_H
#define GET_MCMC_ESTIMATE_H

#include <Eigen/Dense>
#include <string>
#include <iostream> // for cerr
#include <fstream>

#include "param_pack.h"

Eigen::VectorXd ave_mcmc_samps(const std::string& file_loc, int burn_in, const std::vector<TransType>& tts, bool from_trans, bool to_trans);

// TODO handle if certain rows have fewer columns (this was an error)
#endif // GET_MCMC_ESTIMATE_H
