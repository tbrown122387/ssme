#ifndef UTILIS_H
#define UTILIS_H

#include <Eigen/Dense>
#include <fstream>
#include <array>
#include <iostream> // std::cerr
#include <fstream>  // std::ofstream
#include <string>   // string
#include <vector>   // vector


/**
 * @namesapce utils
 */
namespace utils{


    /**
     * @brief return string with format is YYYY-MM-DD.HH:mm:ss
     */
    std::string gen_string_with_time(const std::string& str) {
        time_t     now = time(0);
        struct tm  tstruct;
        char       buf[80];
        tstruct = *localtime(&now);
        strftime(buf, sizeof(buf), "%Y-%m-%d.%H-%M-%S", &tstruct);
        return str + "_" + buf;
    }

    
    /**
     * @brief reads in data in a csv file with no header and separated by commas.
     * @tparam nc the number of columns.
     * @param file_loc the string filepath of the file.
     * @return a std::vector of your data. Each elemenet is a row in Eigen::Vector form.
     */
    template<size_t nc, typename float_t>
    std::vector<Eigen::Matrix<float_t,nc,1> > read_data(const std::string& file_loc)
    {
        // returning this. gotta build it up
        std::vector<Eigen::Matrix<float_t,nc,1> > data;
        
        // start reading
        std::string line;
        std::ifstream ifs(file_loc);
        std::string one_number;    
        if(!ifs.is_open()){     // check if we can open inFile
            std::cerr << "readInData() failed to read data from: " << file_loc << "\n";
        }
        
        // didn't fail...keep going
        while ( std::getline(ifs, line) ){     // get a whole row as a string
        
            std::vector<float_t> data_row;
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
            
            // now append this Vec to your collection
            Eigen::Map<Eigen::Matrix<float_t,nc,1> > drw(&data_row[0], nc);
            data.push_back(drw);
        }
        
        return data;
    } 

} // namespace utils


#endif // UTILIS_H
