#include <ctime>
#include <iomanip> // std::setprecision, std::setw, std::setfill
#include "system/system_parameters.hpp"
#include "cuda/cuda_matrix_base.hpp"
#include "misc/commandline_io.hpp"
#include "misc/escape_sequences.hpp"
#include "misc/timeit.hpp"
#include "omp.h"

// File-Local Configuration
static size_t console_width = 120;
static char seperator = '-';

/**
 * Ugly Function to pretty-print "PULSE" as a colored logo to the cmd
 * Good thing writing code like this is not punishable by law, even
 * though I agree it should be.
 */
void print_name() {
    std::cout << PC3::CLIO::fillLine( console_width, seperator ) << "\n\n"; // Horizontal Seperator
    std::cout << EscapeSequence::BLUE << EscapeSequence::BOLD;              // Make Text Blue and Bold
    // Print Pulse LOGO
    std::cout << PC3::CLIO::centerString( " _____    _     _            _______   _______  ", console_width ) << "\n";
    std::cout << PC3::CLIO::centerString( "|_____]   |     |   |        |______   |______  ", console_width ) << "\n";
    std::cout << PC3::CLIO::centerString( "|       . |_____| . |_____ . ______| . |______ .", console_width ) << "\n\n";
    std::stringstream ss;
    ss << EscapeSequence::RESET << EscapeSequence::UNDERLINE << EscapeSequence::BOLD
       << EscapeSequence::BLUE << "P" << EscapeSequence::GRAY << "aderborn " << EscapeSequence::BLUE
       << "U" << EscapeSequence::GRAY << "ltrafast So" << EscapeSequence::BLUE << "L" << EscapeSequence::GRAY
       << "ver for the nonlinear " << EscapeSequence::BLUE << "S" << EscapeSequence::GRAY << "chroedinger "
       << EscapeSequence::BLUE << "E" << EscapeSequence::GRAY << "quation" << EscapeSequence::RESET;
    std::cout << PC3::CLIO::centerStringRaw( ss.str(), console_width, "Paderborn Ultrafast SoLver for the nonlinear Schroedinger Equation" ) << std::endl;
    ss.str( "" );
    ss.clear();
    ss << "Version: " << EscapeSequence::BOLD << EscapeSequence::BLUE << "0.1.0" << EscapeSequence::RESET;
    std::cout << PC3::CLIO::centerStringRaw( ss.str(), console_width, "Version: 0.1.0" ) << std::endl;
    std::cout << PC3::CLIO::centerString( "https://github.com/davidbauch/PC3", console_width ) << std::endl;
    std::cout << PC3::CLIO::fillLine( console_width, seperator ) << "\n"; // Horizontal Seperator
}

void PC3::SystemParameters::printHelp() {
    print_name();
#ifndef USE_HALF_PRECISION
    std::cout << "This program is compiled with " << EscapeSequence::UNDERLINE << EscapeSequence::YELLOW << "double precision" << EscapeSequence::RESET << " numbers.\n";
#else
    std::cout << "This program is compiled with " << EscapeSequence::UNDERLINE << EscapeSequence::YELLOW << "single precision" << EscapeSequence::RESET << " numbers.\n";
#endif
#ifdef USE_CPU
    std::cout << "This program is compiled as a " << EscapeSequence::UNDERLINE << EscapeSequence::YELLOW << "CPU Version" << EscapeSequence::RESET << ".\n";
    std::cout << "Maximum number of CPU cores utilized " << omp_get_max_threads() << std::endl;
#endif
    std::cout
        << EscapeSequence::BOLD << PC3::CLIO::fillLine( console_width, seperator ) << EscapeSequence::RESET << std::endl
        << PC3::CLIO::unifyLength( "Flag", "Inputs", "Description\n" ) << std::endl
        << PC3::CLIO::unifyLength( "--path", "<string>", "Workingfolder. Standard is '" + filehandler.outputPath + "'" ) << std::endl
        << PC3::CLIO::unifyLength( "--name", "<string>", "File prefix. Standard is '" + filehandler.outputName + "'" ) << std::endl
        //<< PC3::CLIO::unifyLength( "--loadFrom", "<string> <string...>", "Loads list of matrices from path." ) << std::endl
        << PC3::CLIO::unifyLength( "--config", "<string>", "Loads configuration from file." ) << std::endl
        << PC3::CLIO::unifyLength( "--outEvery", "<int>", "Number of Runge-Kutta iterations for each plot. Standard is every " + std::to_string( output_every ) + " ps" ) << std::endl
        << PC3::CLIO::unifyLength( "--output", "<string...>", "Comma seperated list of things to output. Available: mat,scalar,fft,pump,mask,psi,n. Many can also be specified with _plus or _minus." ) << std::endl
        //<< PC3::CLIO::unifyLength( "--history", "<Y> <points>", "Outputs a maximum number of x-slices at Y for history. y-slices are not supported." ) << std::endl
        << PC3::CLIO::unifyLength( "--historyMatrix", "<int> <int> <int> <int> <int>", "Outputs the matrices specified in --output with specified startx,endx,starty,endy index and increment." ) << std::endl
        //<< PC3::CLIO::unifyLength( "--input", "<string...>", "Comma seperated list of things to input. Available: mat,scalar,fft,pump,mask,psi,n. Many can also be specified with _plus or _minus." ) << std::endl
        << PC3::CLIO::unifyLength( "-nosfml", "no arguments", "If passed to the program, disables all live graphical output. " ) << std::endl;
    std::cout << PC3::CLIO::fillLine( console_width, seperator ) << std::endl;
    std::cout << PC3::CLIO::unifyLength( "Numerical parameters", "", "" ) << std::endl
              << PC3::CLIO::unifyLength( "Flag", "Inputs", "Description" ) << std::endl
              << PC3::CLIO::unifyLength( "--N", "<int> <int>", "Grid Dimensions (N x N). Standard is " + std::to_string( p.N_x ) + " x " + std::to_string( p.N_y ) ) << std::endl
              << PC3::CLIO::unifyLength( "--tstep", "<double>", "Timestep, standard is magic-timestep = " + PC3::CLIO::to_str( magic_timestep ) + "ps" ) << std::endl
              << PC3::CLIO::unifyLength( "--iterator", "<string>", "RK4, RK45 or SSFM" ) << std::endl
              << PC3::CLIO::unifyLength( "-rk45", "no arguments", "Use RK45" ) << std::endl
              << PC3::CLIO::unifyLength( "--rk45dt", "<double> <double>", "dt_min and dt_max for the RK45 method" ) << std::endl
              << PC3::CLIO::unifyLength( "-ssfm", "no arguments", "Use SSFM" ) << std::endl
              << PC3::CLIO::unifyLength( "--tol", "<double>", "RK45 Tolerance, standard is " + PC3::CLIO::to_str( tolerance ) + " ps" ) << std::endl
              << PC3::CLIO::unifyLength( "--tmax", "<double>", "Timelimit, standard is " + PC3::CLIO::to_str( t_max ) + " ps" ) << std::endl
              << PC3::CLIO::unifyLength( "--boundary", "<string> <string>", "Boundary conditions for x and y. Is either 'periodic' or 'zero'." ) << std::endl;
    std::cout << PC3::CLIO::fillLine( console_width, seperator ) << std::endl;
    std::cout << PC3::CLIO::unifyLength( "System Parameters", "", "" ) << std::endl
              << PC3::CLIO::unifyLength( "Flag", "Inputs", "Description" ) << std::endl
              << PC3::CLIO::unifyLength( "--gammaC", "<double>", "Standard is " + PC3::CLIO::to_str( p.gamma_c ) + " ps^-1" ) << std::endl
              << PC3::CLIO::unifyLength( "--gammaR", "<double>", "Standard is " + PC3::CLIO::to_str( p.gamma_r / p.gamma_c ) + "*gammaC" ) << std::endl
              << PC3::CLIO::unifyLength( "--gc", "<double>", "Standard is " + PC3::CLIO::to_str( p.g_c ) + " eV mum^2" ) << std::endl
              << PC3::CLIO::unifyLength( "--gr", "<double>", "Standard is " + PC3::CLIO::to_str( p.g_r / p.g_c ) + "*gc" ) << std::endl
              << PC3::CLIO::unifyLength( "--meff", "<double>", "Standard is " + PC3::CLIO::to_str( p.m_eff ) ) << std::endl
              << PC3::CLIO::unifyLength( "--R", "<double>", "Standard is " + PC3::CLIO::to_str( p.R ) + " ps^-1 mum^2" ) << std::endl
              << PC3::CLIO::unifyLength( "--g_pm", "<double>", "Standard is " + PC3::CLIO::to_str( p.g_pm / p.g_c ) + "*gc. Only effective in a system with TE/TM splitting." ) << std::endl
              << PC3::CLIO::unifyLength( "--deltaLT", "<double>", "Standard is " + PC3::CLIO::to_str( p.delta_LT ) + " eV. Only effective in a system with TE/TM splitting." ) << std::endl
              << PC3::CLIO::unifyLength( "--L", "<double> <double>", "Standard is " + PC3::CLIO::to_str( p.L_x ) + ", " + PC3::CLIO::to_str( p.L_y ) + " mum" ) << std::endl;
    std::cout << PC3::CLIO::fillLine( console_width, seperator ) << std::endl;
    std::cout << PC3::CLIO::unifyLength( "Envelopes.", "", "" ) << std::endl
              << PC3::CLIO::unifyLength( "Envelopes are passed using either their spatial and temporal characteristics, or by loading an external file. Syntax:", "", "" ) << std::endl
              << PC3::CLIO::unifyLength( "--envelope", "<double> <string> <double> <double> <double> <double> <string> <double> <double> <string> osc <double> <double> <double>", "amplitude, behaviour (add,multiply,replace,adaptive,complex), widthX, widthY, posX, posY, pol (plus,minus,both), exponent, charge, type (gauss, ring), [t0, frequency, sigma]. 'osc' signals the temporal envelope, which can be omitted for constant envelope." ) << std::endl
              << PC3::CLIO::unifyLength( "--envelope", "<string> <double> <string> <string> osc <double> <double> <double>", "path, amplitude, behaviour (add,multiply,replace,adaptive,complex), pol (plus,minus,both) t0, frequency, sigma" ) << std::endl
              << "Possible Envelopes include:" << std::endl
              << PC3::CLIO::unifyLength( "--pump", "Spatial and Temporal ~cos(wt)", "" ) << std::endl
              << PC3::CLIO::unifyLength( "--potential", "Spatial and Temporal ~cos(wt)", "" ) << std::endl
              << PC3::CLIO::unifyLength( "--initialState", "Spatial", "" ) << std::endl
              << PC3::CLIO::unifyLength( "--initialReservoir", "Spatial", "" ) << std::endl
              << PC3::CLIO::unifyLength( "--pulse", "Spatial and Temporal ~exp(iwt)", "" ) << std::endl
              << PC3::CLIO::unifyLength( "--fftMask", "Spatial", "" ) << std::endl
              << "Additional Parameters:" << std::endl
              << PC3::CLIO::unifyLength( "--fftEvery", "<int>", "Apply FFT Filter every x ps" ) << std::endl
              << PC3::CLIO::unifyLength( "--initRandom", "<double>", "Amplitude. Randomly initialize Psi" ) << std::endl;
    std::cout << PC3::CLIO::fillLine( console_width, seperator ) << std::endl;
    std::cout << PC3::CLIO::unifyLength( "SI Scalings", "", "" ) << std::endl
              << PC3::CLIO::unifyLength( "Flag", "Inputs", "Description" ) << std::endl
              << PC3::CLIO::unifyLength( "--hbar", "<double>", "Standard is " + PC3::CLIO::to_str( p.h_bar ) ) << std::endl
              << PC3::CLIO::unifyLength( "--e", "<double>", "Standard is " + PC3::CLIO::to_str( p.e_e ) ) << std::endl
              << PC3::CLIO::unifyLength( "--me", "<double>", "Standard is " + PC3::CLIO::to_str( p.m_e ) ) << std::endl
              << PC3::CLIO::unifyLength( "--hbarscaled", "<double>", "Standard is " + PC3::CLIO::to_str( p.h_bar_s ) ) << std::endl
              << PC3::CLIO::unifyLength( "--meff", "<double>", "Standard is " + PC3::CLIO::to_str( p.m_eff ) ) << std::endl;
#ifdef USE_CPU
    std::cout << PC3::CLIO::unifyLength( "--threads", "<int>", "Standard is " + std::to_string( omp_max_threads ) + " Threads\n" ) << std::endl;
#endif
}

void PC3::SystemParameters::printSummary( std::map<std::string, std::vector<double>> timeit_times, std::map<std::string, double> timeit_times_total ) {
    print_name();
    const int l = 15;
    std::cout << EscapeSequence::BOLD << PC3::CLIO::fillLine( console_width, seperator ) << EscapeSequence::RESET << std::endl;
    std::cout << EscapeSequence::BOLD << PC3::CLIO::centerString( " Parameters ", console_width, '-' ) << EscapeSequence::RESET << std::endl;
    std::cout << PC3::CLIO::unifyLength( "N", std::to_string( p.N_x ) + ", " + std::to_string( p.N_y ), "", l, l, l, " " ) << std::endl;
    std::cout << PC3::CLIO::unifyLength( "N^2", std::to_string( p.N_x * p.N_y ), "", l, l, l, " " ) << std::endl;
    std::cout << PC3::CLIO::unifyLength( "Lx", PC3::CLIO::to_str( p.L_x ), "mum", l, l, l, " " ) << std::endl;
    std::cout << PC3::CLIO::unifyLength( "Ly", PC3::CLIO::to_str( p.L_y ), "mum", l, l, l, " " ) << std::endl;
    std::cout << PC3::CLIO::unifyLength( "dx", PC3::CLIO::to_str( p.dx ), "mum", l, l, l, " " ) << std::endl;
    std::cout << PC3::CLIO::unifyLength( "dy", PC3::CLIO::to_str( p.dx ), "mum", l, l, l, " " ) << std::endl;
    std::cout << PC3::CLIO::unifyLength( "tmax", PC3::CLIO::to_str( t_max ), "ps", l, l, l, " " ) << std::endl;
    std::cout << PC3::CLIO::unifyLength( "dt", PC3::CLIO::to_str( p.dt ), "ps", l, l, l, " " ) << std::endl;
    std::cout << PC3::CLIO::unifyLength( "gamma_c", PC3::CLIO::to_str( p.gamma_c ), "ps^-1", l, l, l, " " ) << std::endl;
    std::cout << PC3::CLIO::unifyLength( "gamma_r", PC3::CLIO::to_str( p.gamma_r ), "ps^-1", l, l, l, " " ) << std::endl;
    std::cout << PC3::CLIO::unifyLength( "g_c", PC3::CLIO::to_str( p.g_c ), "eV mum^2", l, l, l, " " ) << std::endl;
    std::cout << PC3::CLIO::unifyLength( "g_r", PC3::CLIO::to_str( p.g_r ), "eV mum^2", l, l, l, " " ) << std::endl;
    std::cout << PC3::CLIO::unifyLength( "g_pm", PC3::CLIO::to_str( p.g_pm ), "eV mum^2", l, l, l, " " ) << std::endl;
    std::cout << PC3::CLIO::unifyLength( "R", PC3::CLIO::to_str( p.R ), "ps^-1 mum^-2", l, l, l, " " ) << std::endl;
    std::cout << PC3::CLIO::unifyLength( "delta_LT", PC3::CLIO::to_str( p.delta_LT ), "eV", l, l, l, " " ) << std::endl;
    std::cout << PC3::CLIO::unifyLength( "m_eff", PC3::CLIO::to_str( p.m_eff ), "", l, l, l, " " ) << std::endl;
    std::cout << PC3::CLIO::unifyLength( "h_bar_s", PC3::CLIO::to_str( p.h_bar_s ), "", l, l, l, " " ) << std::endl;
    std::cout << "Boundary Condition: " << ( p.periodic_boundary_x ? "Periodic" : "Zero" ) << "(x):" << ( p.periodic_boundary_y ? "Periodic" : "Zero" ) << "(y)" << std::endl;
    std::cout << PC3::CLIO::centerString( " Envelope Functions ", console_width, '-' ) << std::endl;
    // TODO: overwrite << operator of the Envelope Class
    if ( pulse.size() > 0 )
        std::cout << "Pulse Envelopes:\n"
                  << pulse.toString();
    if ( pump.size() > 0 )
        std::cout << "Pump Envelopes:\n"
                  << pump.toString();
    if ( potential.size() > 0 )
        std::cout << "Potential Envelopes:\n"
                  << potential.toString();
    if ( fft_mask.size() > 0 )
        std::cout << "FFT Mask Envelopes:\n"
                  << fft_mask.toString();
    if ( initial_state.size() > 0 )
        std::cout << "Initial State Envelopes:\n"
                  << initial_state.toString();
    std::cout << EscapeSequence::BOLD << PC3::CLIO::centerString( " Runtime Statistics ", console_width, '-' ) << EscapeSequence::RESET << std::endl;
    double total = PC3::TimeIt::totalRuntime();
    std::cout << "Total Runtime: " << total << "s --> " << ( total / p.t * 1E3 ) << "ms/ps --> " << ( p.t / total ) << "ps/s --> " << ( total / iteration )*1000000.0 << " mus/it" << std::endl;
    std::cout << EscapeSequence::BOLD << PC3::CLIO::centerString( " Infos ", console_width, '-' ) << EscapeSequence::RESET << std::endl;
    
    std::cout << "Calculations done using the '" << iterator << "' solver" << std::endl;
    if ( iterator == "rk45" ) {
        std::cout << " = Tolerance used: " << tolerance << std::endl;
        std::cout << " = dt_max used: " << dt_max << std::endl;
        std::cout << " = dt_min used: " << dt_min << std::endl;
    }

    std::cout << "Calculated until t = " << p.t << "ps" << std::endl;
    if ( fft_mask.size() > 0 )
        std::cout << "Applying FFT every " << fft_every << " ps" << std::endl;
    std::cout << "Output variables and plots every " << output_every << " ps" << std::endl;
    std::cout << "Total allocated space for Device Matrices: " << CUDAMatrixBase::global_total_device_mb_max << " MB." << std::endl;
    std::cout << "Total allocated space for Host Matrices: " << CUDAMatrixBase::global_total_host_mb_max << " MB." << std::endl;
    std::cout << "Random Seed was: " << random_seed << std::endl;
#ifdef USE_HALF_PRECISION
    std::cout << "This program is compiled using " << EscapeSequence::UNDERLINE << EscapeSequence::BLUE << "single precision" << EscapeSequence::RESET << " numbers.\n";
#else
    std::cout << "This program is compiled using " << EscapeSequence::UNDERLINE << EscapeSequence::BLUE << "double precision" << EscapeSequence::RESET << " numbers.\n";
#endif
#ifdef USE_CPU
    std::cout << "Device Used: " << EscapeSequence::BOLD << EscapeSequence::YELLOW << "CPU" << EscapeSequence::RESET << std::endl;
    std::cout << EscapeSequence::GRAY << "  CPU cores utilized: " << omp_max_threads << EscapeSequence::RESET << std::endl;
#else
// The Headers required for this come from system_parameters.hpp->typedef.cuh
    int nDevices;
    cudaGetDeviceCount( &nDevices ); 
    int device;
    cudaGetDevice( &device );
    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, device );
    std::cout << "Device Used: " << EscapeSequence::GREEN << EscapeSequence::BOLD << prop.name << EscapeSequence::RESET << std::endl;
    std::cout << EscapeSequence::GRAY << "  Memory Clock Rate (GHz): " << prop.memoryClockRate / 1024.0 / 1024.0 << std::endl;
    std::cout << "  Peak Memory Bandwidth (GB/s): " << 2.0 * prop.memoryClockRate * ( prop.memoryBusWidth / 8 ) / 1.0e6 << std::endl;
    std::cout << "  Total global memory (GB): " << (float)( prop.totalGlobalMem ) / 1024.0 / 1024.0 / 1024.0 << std::endl;
    std::cout << "  Warp-size: " << prop.warpSize << std::endl;
    std::cout << "  CUDA Cores: " << prop.multiProcessorCount * _ConvertSMVer2Cores( prop.major, prop.minor ) << std::endl;
    std::cout << "  GPU Clock Rate (GHz): " << prop.clockRate / 1024.0 / 1024.0 << EscapeSequence::RESET << std::endl;
#endif
    std::cout << EscapeSequence::BOLD << PC3::CLIO::fillLine( console_width, '=' ) << EscapeSequence::RESET << std::endl;
}

double _pc3_last_output_time = 0.;

void PC3::SystemParameters::printCMD( double complete_duration, double complete_iterations ) {
    // TODO: move this into function and hide the ugly thing where noone can find it.
    if ( std::time( nullptr ) - _pc3_last_output_time < 0.25 )
        return;
    // Print Runtime
    std::cout << EscapeSequence::HIDE_CURSOR;
    std::cout << PC3::CLIO::fillLine( console_width, seperator ) << std::endl;
    std::cout << "    T = " << int( p.t ) << "ps - dt = " << std::setprecision( 2 ) << p.dt << "ps    \n";
    // Progressbar for p.t/t_max
    std::cout << "    Progress:  " << PC3::CLIO::createProgressBar( p.t, t_max, console_width - 30 ) << "    \n";
    bool evaluate_stochastic = evaluateStochastic();
    std::cout << "    Current System: " << ( p.use_twin_mode ? "TE/TM" : "Scalar" ) << " - " << ( evaluate_stochastic ? "With Stochastic" : "No Stochastic" ) << "    \n";
    std::cout << "    Runtime: " << int( complete_duration ) << "s, remaining: " << int( complete_duration * ( t_max - p.t ) / p.t ) << "s    \n";
    std::cout << "    Time per ps: " << complete_duration / p.t << "s/ps  -  " << std::setprecision( 3 ) << p.t / complete_duration << "ps/s  -  " << complete_iterations / complete_duration << "it/s    \n";
    std::cout << PC3::CLIO::fillLine( console_width, seperator ) << std::endl;
    std::cout << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP;
    _pc3_last_output_time = std::time( nullptr );
}

void PC3::SystemParameters::finishCMD() {
    std::cout << "\n\n\n\n\n\n\n"
              << EscapeSequence::SHOW_CURSOR;
}
