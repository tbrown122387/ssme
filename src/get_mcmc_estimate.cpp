#include "get_mcmc_estimate.h"

Eigen::VectorXd ave_mcmc_samps(const std::string& file_loc, int burn_in, const std::vector<TransType>& tts, bool from_trans, bool to_trans)
{
    // start reading
    std::string line;
    std::ifstream ifs(file_loc);
    std::string one_number;    
    if(!ifs.is_open()){     // check if we can open inFile
        std::cerr << "ave_mcmc_samps() failed to read data from: " << file_loc << "\n";
    }
    
    // read through all the lines
    Eigen::VectorXd ave_sample, one_row_eigen; 
    int obs_num = 1;
    double n;
    while ( std::getline(ifs, line) ){     // get a whole row as a string
        
        std::vector<double> data_row;
        try{
            // get a single element on a row
            std::istringstream buff(line);
            
            // make one number between commas
            while(std::getline(buff, one_number, ',')){ 
                data_row.push_back(std::stod(one_number));
            }

        } catch (const std::invalid_argument& ia){
            std::cerr << "Invalid Argument: " << ia.what() << "\n";
            continue;
        }   
        
	    if( obs_num - burn_in == 1){

	        Eigen::Map<Eigen::VectorXd> one_row_eigen(&data_row[0], data_row.size());
	        ave_sample = one_row_eigen;
            
	    }else if( obs_num - burn_in > 1){

            Eigen::Map<Eigen::VectorXd> one_row_eigen(&data_row[0], data_row.size());
            n = (double)(obs_num - burn_in);
            ave_sample = (ave_sample*(n-1) + one_row_eigen) / n;
        }
        obs_num++;
    }
    

    if(to_trans){
        return paramPack(ave_sample, tts, from_trans).getTransParams();
    }else{
        return paramPack(ave_sample, tts, from_trans).getUnTransParams();
    }

} 



