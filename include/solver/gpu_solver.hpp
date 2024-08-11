#pragma once

#include <iostream>
#include <map>
#include <functional>
#include "cuda/typedef.cuh"
#include "cuda/cuda_matrix.cuh"
#include "cuda/cuda_macro.cuh"
#include "kernel/kernel_fft.cuh"
#include "system/system_parameters.hpp"
#include "system/filehandler.hpp"
#include "solver/matrix_container.hpp"
#include "misc/escape_sequences.hpp"

namespace PC3 {

#define INSTANCIATE_K( index, twin )                                                                           \
    matrix.k##index##_wavefunction_plus.constructDevice( system.N_x, system.N_y, "matrix.k" #index "_wavefunction_plus" );       \
    matrix.k##index##_reservoir_plus.constructDevice( system.N_x, system.N_y, "matrix.k" #index "_reservoir_plus" );             \
    if ( twin ) {                                                                                              \
        matrix.k##index##_wavefunction_minus.constructDevice( system.N_x, system.N_y, "matrix.k" #index "_wavefunction_minus" ); \
        matrix.k##index##_reservoir_minus.constructDevice( system.N_x, system.N_y, "matrix.k" #index "_reservoir_minus" );       \
    }

/**
 * @brief GPU Solver class providing the interface for the GPU solver.
 * Implements RK4, RK45, FFT calculations.
 *
 */
class Solver {
   public:
    // References to system and filehandler so we dont need to pass them around all the time
    PC3::SystemParameters& system;
    PC3::FileHandler& filehandler;

    // TODO: remove
    struct TemporalEvelope {
        PC3::CUDAMatrix<Type::complex> amp;

        struct Pointers {
            Type::complex* amp;
            unsigned int n;
        };

        Pointers pointers() {
            return Pointers{ amp.getDevicePtr(), amp.getTotalSize() };
        }
    } dev_pulse_oscillation, dev_pump_oscillation, dev_potential_oscillation;

    // Device Variables
    MatrixContainer matrix;

    // Cache Maps
    std::map<std::string, std::vector<Type::real>> cache_map_scalar;

    Solver( PC3::SystemParameters& system ) : system( system ), filehandler( system.filehandler ) {
        std::cout << PC3::CLIO::prettyPrint( "Creating Solver...", PC3::CLIO::Control::Info ) << std::endl;

        // Initialize all host matrices
        initializeHostMatricesFromSystem();
        // Then output all matrices to file. If --output was not passed in argv, this method outputs everything.
        outputInitialMatrices();
        // Copy remaining stuff to Device.
        initializeDeviceMatricesFromHost();

    }
    
    void initializeHostMatricesFromSystem();               // Evaluates the envelopes and initializes the host matrices
    void initializeDeviceMatricesFromHost();               // Transfers the host matrices to their device equivalents

    // Output (Final) Host Matrices to files
    void outputMatrices( const unsigned int start_x, const unsigned int end_x, const unsigned int start_y, const unsigned int end_y, const unsigned int increment, const std::string& suffix = "", const std::string& prefix = "" );
    // Output Initial Host Matrices to files
    void outputInitialMatrices();

    // Output the history and max caches to files. should be called from finalize()
    void cacheToFiles();

    void finalize();

    void iterateFixedTimestepRungeKutta3( dim3 block_size, dim3 grid_size );
    void iterateFixedTimestepRungeKutta4( dim3 block_size, dim3 grid_size );
    void iterateVariableTimestepRungeKutta( dim3 block_size, dim3 grid_size );
    void iterateSplitStepFourier( dim3 block_size, dim3 grid_size );
    void iterateImaginaryTimePropagation( dim3 block_size, dim3 grid_size );

    struct iteratorFunction {
        int k_max;
        std::function<void( dim3, dim3 )> iterate;
    };
    std::map<std::string, iteratorFunction> iterator = {
        { "rk3", { 3, std::bind( &Solver::iterateFixedTimestepRungeKutta3, this, std::placeholders::_1, std::placeholders::_2 ) } },
        { "rk4", { 4, std::bind( &Solver::iterateFixedTimestepRungeKutta4, this, std::placeholders::_1, std::placeholders::_2 ) } },
        { "rk45", { 6, std::bind( &Solver::iterateVariableTimestepRungeKutta, this, std::placeholders::_1, std::placeholders::_2 ) } },
        { "ssfm", { 2, std::bind( &Solver::iterateSplitStepFourier, this, std::placeholders::_1, std::placeholders::_2 ) } },
        { "itp", { 1, std::bind( &Solver::iterateImaginaryTimePropagation, this, std::placeholders::_1, std::placeholders::_2 ) } }
    };
    void sum_block_1(int N,Type::real* __restrict A,Type::real alpha,Type::real* __restrict B,Type::real* __restrict C);
    void sum_block_4(int N,Type::real* __restrict A,Type::real alpha1,Type::real* __restrict B1,Type::real alpha2,Type::real* __restrict B2,Type::real alpha3,Type::real* __restrict B3,Type::real alpha4,Type::real* __restrict B4,Type::real* __restrict C);    void gp_scalar2(int Nx,int Ny,Type::real t,Type::real m2_over_dx2_p_dy2,Type::real one_over_dx2,Type::real one_over_dy2,Type::real minus_1_over_h_bar_s,Type::real g_c,Type::real g_r,Type::real R,Type::real gamma_c,Type::real gamma_r,Type::real m_eff_scaled,Type::real* __restrict in_wfu_real,Type::real* __restrict in_wfu_imag,Type::real* __restrict in_res_real,Type::real* __restrict in_res_imag,Type::real* __restrict out_wfu_real,Type::real* __restrict out_wfu_imag,Type::real* __restrict out_res_real,Type::real* __restrict out_res_imag);
    bool iterate();

    void applyFFTFilter( dim3 block_size, dim3 grid_size, bool apply_mask = true );

    enum class FFT {
        inverse,
        forward
    };
    void calculateFFT( Type::complex* device_ptr_in, Type::complex* device_ptr_out, FFT dir );

    void swapBuffers();

    void cacheValues();
    void cacheMatrices();
};

// Helper macro to choose the correct runge function
//#define RUNGE_FUNCTION_GP (p.use_twin_mode ? PC3::Kernel::Compute::gp_tetm : PC3::Kernel::Compute::gp_scalar)
#define RUNGE_FUNCTION_GP PC3::Kernel::Compute::gp_scalar

// Helper Macro to iterate a specific RK K
#define CALCULATE_K( index, time, input_wavefunction, input_reservoir ) \
CALL_KERNEL( \
    RUNGE_FUNCTION_GP, "K"#index, grid_size, block_size,  \
    time, device_pointers, p, pulse_pointers, pump_pointers, potential_pointers, \
    {  \
        device_pointers.input_wavefunction##_plus, device_pointers.input_wavefunction##_minus, device_pointers.input_reservoir##_plus, device_pointers.input_reservoir##_minus, \
        device_pointers.k##index##_wavefunction_plus, device_pointers.k##index##_wavefunction_minus, device_pointers.k##index##_reservoir_plus, device_pointers.k##index##_reservoir_minus \
    } \
);


} // namespace PC3
