#include "estimate_univ_svol.h"

#include <iostream> // std::cerr
#include <sstream>  // std::stringstream
#include <string>   // std::string
#include <cstdlib>  // atoi()


#define NUMPARTS 500
#define DIMOBS 1
#define NUMPARAMS 3
#define DIMSTATE 1
#define FLOATTYPE float

int main(int argc, char* argv[]){

    // read in the arguments from the command line
    std::string procedure;
    if(argc != 6){
        std::cerr << "Please enter:\n"
                                                    "1.) datafile location, \n"
                                                    "2.) samples_base_name, \n"
                                                    "3.) messages file base name, \n"
                                                    "4.) number of mcmc iterations. \n"
                                                    "5.) number of pfilters. \n";
        return 0;
    }


   // get arguments and call function
   std::string data_loc = argv[1];
   std::string samples_base_name = argv[2];
   std::string messages_base_name = argv[3];
   unsigned int num_mcmc_iters = atoi(argv[4]);
   unsigned int num_pfilters = atoi(argv[5]);
   do_ada_pmmh_univ_svol<NUMPARAMS,DIMSTATE,DIMOBS,NUMPARTS,FLOATTYPE>(
		   					                                 data_loc, 
                                                             samples_base_name, 
                                                             messages_base_name, 
                                                             num_mcmc_iters, 
                                                             num_pfilters,
                                                             false); // use multicore?
   
   
   return 0;
}
