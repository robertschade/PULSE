#ifndef USECPU
#include <cuComplex.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#else
#include <numeric>
#endif

#include <complex>
#include <omp.h>

// Include Cuda Kernel headers
#include "cuda/cuda_complex.cuh"
#include "cuda/cuda_macro.cuh"
#include "kernel/kernel_runge_function.cuh"
#include "kernel/kernel_summation.cuh"
#include "kernel/kernel_fft.cuh"
#include "system/system.hpp"
#include "misc/helperfunctions.hpp"
#include "cuda/cuda_matrix.cuh"
#include "solver/gpu_solver.hpp"
#include "kernel/kernel_random_numbers.cuh"

/*
 * Helper variable for caching the current time for FFT evaluations.
 * We dont need this variable anywhere else, so we just create it
 * locally to this file here.
 */
real_number cached_t = 0.0;

// Helper macro to choose the correct runge function
#define RUNGE_FUNCTION_GP (p.use_twin_mode ? PC3::Kernel::Compute::gp_tetm : PC3::Kernel::Compute::gp_scalar)

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

/*
 * This function iterates the Runge Kutta Kernel using a fixed time step.
 * A 4th order Runge-Kutta method is used. This function calls a single
 * rungeFuncSum function with varying delta-t. Calculation of the inputs
 * for the next rungeFuncKernel call is done in the rungeFuncSum function.
 * The general implementation of the RK4 method goes as follows:
 * ------------------------------------------------------------------------------
 * k1 = f(t, y) = rungeFuncKernel(current)
 * input_for_k2 = current + 0.5 * dt * k1
 * k2 = f(t + 0.5 * dt, input_for_k2) = rungeFuncKernel(input_for_k2)
 * input_for_k3 = current + 0.5 * dt * k2
 * k3 = f(t + 0.5 * dt, input_for_k3) = rungeFuncKernel(input_for_k3)
 * input_for_k4 = current + dt * k3
 * k4 = f(t + dt, input_for_k4) = rungeFuncKernel(input_for_k4)
 * next = current + dt * (1/6 * k1 + 1/3 * k2 + 1/3 * k3 + 1/6 * k4)
 * ------------------------------------------------------------------------------
 * The Runge method iterates psi,k1-k4 to psi_next using a wave-like approach.
 * We calculate 4 rows of k1, 3 rows of k2, 2 rows of k3 and 1 row of k4 before the first iteration.
 * Then, we iterate all of the remaining rows after each other, incrementing the buffer for the next iteration.
 */

void PC3::Solver::iterateFixedTimestepRungeKutta( dim3 block_size, dim3 grid_size ) {
    // This variable contains all the system parameters the kernel could need
    auto p = system.kernel_parameters;
    
    // This variable contains all the device pointers the kernel could need
    auto device_pointers = matrix.pointers();

    // The CPU should briefly evaluate wether the pulse and the reservoir have to be evaluated
    bool evaluate_pulse = system.evaluatePulse();
    bool evaluate_reservoir = system.evaluateReservoir();
    bool evaluate_stochastic = system.evaluateStochastic();

    // Pointers to Oscillation Parameters
    auto pulse_pointers = dev_pulse_oscillation.pointers();
    auto pump_pointers = dev_pump_oscillation.pointers();
    auto potential_pointers = dev_potential_oscillation.pointers();

    // The delta time is either real or imaginary, depending on the system configuration
    complex_number delta_time = system.imaginary_time ? complex_number(0.0, -p.dt) : complex_number(p.dt, 0.0);

    // If required, calculate new set of random numbers.
    if (evaluate_stochastic)
    CALL_KERNEL(
        PC3::Kernel::generate_random_numbers, "random_number_gen", grid_size, block_size,
        device_pointers.random_state, device_pointers.random_number, p.N_x*p.N_y, system.p.stochastic_amplitude*PC3::CUDA::sqrt(p.dt), system.p.stochastic_amplitude*PC3::CUDA::sqrt(p.dt)
    );

    CALCULATE_K( 1, p.t, wavefunction, reservoir );

    CALL_KERNEL(
        Kernel::RK4::runge_sum_to_input_k2, "Sum for K2", grid_size, block_size,
        delta_time, device_pointers, p
    );

    CALCULATE_K( 2, p.t + 0.5 * p.dt, buffer_wavefunction, buffer_reservoir );

    CALL_KERNEL(
        Kernel::RK4::runge_sum_to_input_k3, "Sum for K3", grid_size, block_size,
        delta_time, device_pointers, p
    );

    CALCULATE_K( 3, p.t + 0.5 * p.dt, buffer_wavefunction, buffer_reservoir);

    CALL_KERNEL(
        Kernel::RK4::runge_sum_to_input_k4, "Sum for K4", grid_size, block_size,
        delta_time, device_pointers, p
    );

    CALCULATE_K( 4, p.t + p.dt, buffer_wavefunction, buffer_reservoir);

    CALL_KERNEL(
        Kernel::RK4::runge_sum_to_final, "Final Sum", grid_size, block_size,
        delta_time, device_pointers, p
    );

    // Do one device synchronization to make sure that the kernel has finished
    CHECK_CUDA_ERROR( cudaDeviceSynchronize(), "Sync" );

    normalizeImaginaryTimePropagation(device_pointers, p, block_size, grid_size);

    // Swap the next and current wavefunction buffers. This only swaps the pointers, not the data.
    swapBuffers();
    
    return;
}

struct square_reduction
{
    CUDA_HOST_DEVICE real_number operator()(const complex_number& x) const { 
        const real_number res = PC3::CUDA::abs2(x);
        return res; 
    }
};


/*
* This function iterates the Runge Kutta Kernel using a variable time step.
* A 4th order Runge-Kutta method is used to calculate
* the next y iteration; a 5th order solution is
* used to calculate the iteration error.
* This function calls multiple different rungeFuncSum functions with varying
* delta-t and coefficients. Calculation of the inputs for the next
* rungeFuncKernel call is done in the rungeFuncSum function.
* The general implementation of the RK45 method goes as follows:
* ------------------------------------------------------------------------------
* k1 = f(t, y) = rungeFuncKernel(current)
* input_for_k2 = current + b11 * dt * k1
* k2 = f(t + a2 * dt, input_for_k2) = rungeFuncKernel(input_for_k2)
* input_for_k3 = current + b21 * dt * k1 + b22 * dt * k2
* k3 = f(t + a3 * dt, input_for_k3) = rungeFuncKernel(input_for_k3)
* input_for_k4 = current + b31 * dt * k1 + b32 * dt * k2 + b33 * dt * k3
* k4 = f(t + a4 * dt, input_for_k4) = rungeFuncKernel(input_for_k4)
* input_for_k5 = current + b41 * dt * k1 + b42 * dt * k2 + b43 * dt * k3
                 + b44 * dt * k4
* k5 = f(t + a5 * dt, input_for_k5) = rungeFuncKernel(input_for_k5)
* input_for_k6 = current + b51 * dt * k1 + b52 * dt * k2 + b53 * dt * k3
                 + b54 * dt * k4 + b55 * dt * k5
* k6 = f(t + a6 * dt, input_for_k6) = rungeFuncKernel(input_for_k6)
* next = current + dt * (b61 * k1 + b63 * k3 + b64 * k4 + b65 * k5 + b66 * k6)
* k7 = f(t + a7 * dt, next) = rungeFuncKernel(next)
* error = dt * (e1 * k1 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6 + e7 * k7)
* ------------------------------------------------------------------------------
* The error is then used to update the timestep; If the error is below threshold,
* the iteration is accepted and the total time is increased by dt. If the error
* is above threshold, the iteration is rejected and the timestep is decreased.
* The timestep is always bounded by dt_min and dt_max and will only increase
* using whole multiples of dt_min.
* ------------------------------------------------------------------------------
* @param system The system to iterate
* @param evaluate_pulse If true, the pulse is evaluated at the current time step
*/
void PC3::Solver::iterateVariableTimestepRungeKutta( dim3 block_size, dim3 grid_size ) {
    // Accept current step?
    bool accept = false;
    // This variable contains all the device pointers the kernel could need
    auto device_pointers = matrix.pointers();
    // Pointers to Oscillation Parameters
    auto pulse_pointers = dev_pulse_oscillation.pointers();
    auto pump_pointers = dev_pump_oscillation.pointers();
    auto potential_pointers = dev_potential_oscillation.pointers();
    // The CPU should briefly evaluate wether the pulse and the reservoir have to be evaluated
    bool evaluate_pulse = system.evaluatePulse();
    bool evaluate_reservoir = system.evaluateReservoir();
    bool evaluate_stochastic = system.evaluateStochastic();

    // The delta time is either real or imaginary, depending on the system configuration
    complex_number delta_time = system.imaginary_time ? complex_number(0.0, -system.p.dt) : complex_number(system.p.dt, 0.0);

    // If required, calculate new set of random numbers.
    if (evaluate_stochastic)
    CALL_KERNEL(
        PC3::Kernel::generate_random_numbers, "random_number_gen", grid_size, block_size,
        device_pointers.random_state, device_pointers.random_number, system.p.N_x*system.p.N_y, system.p.stochastic_amplitude*PC3::CUDA::sqrt(system.p.dt), system.p.stochastic_amplitude*PC3::CUDA::sqrt(system.p.dt)
    );

    do {
        // We snapshot here to make sure that the dt is updated
        auto p = system.kernel_parameters;

        CALCULATE_K( 1, p.t, wavefunction, reservoir );

        CALL_KERNEL(
            PC3::Kernel::RK45::runge_sum_to_input_of_k2, "Sum for K2", grid_size, block_size, 
            delta_time, device_pointers, p
        );

        CALCULATE_K( 2, p.t + RKCoefficients::a2 * p.dt, buffer_wavefunction, buffer_reservoir );

        CALL_KERNEL(
            PC3::Kernel::RK45::runge_sum_to_input_of_k3, "Sum for K3", grid_size, block_size, 
            delta_time, device_pointers, p
        );


        CALCULATE_K( 3, p.t + RKCoefficients::a3 * p.dt, buffer_wavefunction, buffer_reservoir );

        CALL_KERNEL(
            PC3::Kernel::RK45::runge_sum_to_input_of_k4, "Sum for K4", grid_size, block_size, 
            delta_time, device_pointers, p
        );

        CALCULATE_K( 4, p.t + RKCoefficients::a4 * p.dt, buffer_wavefunction, buffer_reservoir );

        CALL_KERNEL(
            PC3::Kernel::RK45::runge_sum_to_input_of_k5, "Sum for K5", grid_size, block_size, 
            delta_time, device_pointers, p
        );

        CALCULATE_K( 5, p.t + RKCoefficients::a5 * p.dt, buffer_wavefunction, buffer_reservoir );

        CALL_KERNEL(
            PC3::Kernel::RK45::runge_sum_to_input_of_k6, "Sum for K6", grid_size, block_size, 
            delta_time, device_pointers, p
        );

        CALCULATE_K( 6, p.t + RKCoefficients::a6 * p.dt, buffer_wavefunction, buffer_reservoir );

        // Final Result is in the buffer_ arrays
        CALL_KERNEL(
            PC3::Kernel::RK45::runge_sum_to_final, "Final Sum", grid_size, block_size, 
            delta_time, device_pointers, p
        );

        CALCULATE_K( 7, p.t + RKCoefficients::a7 * p.dt, buffer_wavefunction, buffer_reservoir );

        CALL_KERNEL(
            PC3::Kernel::RK45::runge_sum_final_error, "Final Sum Error", grid_size, block_size, 
            delta_time, device_pointers, p
        );

        // Do one device synchronization to make sure that the kernel has finished
        CHECK_CUDA_ERROR( cudaDeviceSynchronize(), "Sync" );

        #ifndef USECPU
        real_number final_error = thrust::reduce( THRUST_DEVICE, matrix.rk_error.getDevicePtr(), matrix.rk_error.getDevicePtr() + p.N_x * p.N_y, 0.0, thrust::plus<real_number>() );
        real_number sum_abs2 = thrust::transform_reduce( THRUST_DEVICE, matrix.wavefunction_plus.getDevicePtr(), matrix.wavefunction_plus.getDevicePtr() + p.N_x * p.N_y, square_reduction(), 0.0, thrust::plus<real_number>() );
        CHECK_CUDA_ERROR( cudaDeviceSynchronize(), "Sync" );
        #else
        real_number final_error = std::reduce( matrix.rk_error.getDevicePtr(), matrix.rk_error.getDevicePtr() + p.N_x * p.N_y, 0.0, std::plus<real_number>() );
        real_number sum_abs2 = std::transform_reduce( matrix.wavefunction_plus.getDevicePtr(), matrix.wavefunction_plus.getDevicePtr() + p.N_x * p.N_y, 0.0, std::plus<real_number>(), square_reduction() );
        #endif

        // TODO: maybe go back to using max since thats faster
        //auto plus_max = std::get<1>( minmax( matrix.wavefunction_plus.getDevicePtr(), p.N_x * p.N_y, true /*Device Pointer*/ ) );
        final_error = final_error / sum_abs2;

        // Calculate dh
        real_number dh = std::pow( system.tolerance / 2. / CUDA::max( final_error, 1E-15 ), 0.25 );
        // Check if dh is nan
        if ( std::isnan( dh ) ) {
            dh = 1.0;
        }
        if ( std::isnan( final_error ) )
            dh = 0.5;
        
        //  Set new timestep
        system.p.dt = CUDA::min(p.dt * dh, system.dt_max);
        if ( dh < 1.0 )
           system.p.dt = CUDA::max( p.dt - system.dt_min * CUDA::floor( 1.0 / dh ), system.dt_min );
           //p.dt -= system.dt_min;
        else
           system.p.dt = CUDA::min( p.dt + system.dt_min * CUDA::floor( dh ), system.dt_max );
           //p.dt += system.dt_min;

        // Make sure to also update dt from p
        p.dt = p.dt;

        normalizeImaginaryTimePropagation(device_pointers, p, block_size, grid_size);

        // Accept step if error is below tolerance
        if ( final_error < system.tolerance ) {
            accept = true;
            // Swap the next and current wavefunction buffers. This only swaps the pointers, not the data.
            swapBuffers();
        }
    } while ( !accept );
}

/*
 * This function calculates the Fast Fourier Transformation of Psi+ and Psi-
 * and saves the result in dev_fft_plus and dev_fft_minus. These values can
 * then be grabbed using the getDeviceArrays() function. The FFT is then
 * shifted such that k = 0 is in the center of the FFT matrix. Then, the
 * FFT Filter is applied to the FFT, and the FFT is shifted back. Finally,
 * the inverse FFT is calculated and the result is saved in dev_current_Psi_Plus
 * and dev_current_Psi_Minus. The FFT Arrays are shifted once again for
 * visualization purposes.
 * NOTE/OPTIMIZATION: The Shift->Filter->Shift function will be changed later
 * to a cached filter mask, which itself will be shifted.
 */
void PC3::Solver::applyFFTFilter( dim3 block_size, dim3 grid_size, bool apply_mask ) {
    #ifndef USECPU
    // Calculate the actual FFTs
    CHECK_CUDA_ERROR( FFTSOLVER( plan, (fft_complex_number*)matrix.wavefunction_plus.getDevicePtr(), (fft_complex_number*)matrix.fft_plus.getDevicePtr(), CUFFT_FORWARD ), "FFT Exec" );


    // For now, we shift, transform, shift the results. TODO: Move this into one function without shifting
    // Shift FFT to center k = 0
    fft_shift_2D<<<grid_size, block_size>>>( matrix.fft_plus.getDevicePtr(), system.p.N_x, system.p.N_y );
    CHECK_CUDA_ERROR( {}, "FFT Shift Plus" );

    // Do the FFT and the shifting here already for visualization only
    if ( system.p.use_twin_mode ) {
        CHECK_CUDA_ERROR( FFTSOLVER( plan, (fft_complex_number*)matrix.wavefunction_minus.getDevicePtr(), (fft_complex_number*)matrix.fft_minus.getDevicePtr(), CUFFT_FORWARD ), "FFT Exec" );
        fft_shift_2D<<<grid_size, block_size>>>( matrix.fft_minus.getDevicePtr(), system.p.N_x, system.p.N_y );
        CHECK_CUDA_ERROR( {}, "FFT Shift Minus" );
    }
    
    if (not apply_mask)
        return;
    
    // Apply the FFT Mask Filter
    kernel_mask_fft<<<grid_size, block_size>>>( matrix.fft_plus.getDevicePtr(), matrix.fft_mask_plus.getDevicePtr(), system.p.N_x*system.p.N_y );
    CHECK_CUDA_ERROR( {}, "FFT Filter" )
    
    // Undo the shift
    fft_shift_2D<<<grid_size, block_size>>>( matrix.fft_plus.getDevicePtr(), system.p.N_x, system.p.N_y );
    CHECK_CUDA_ERROR( {}, "FFT Shift" )

    // Transform back.
    CHECK_CUDA_ERROR( FFTSOLVER( plan, matrix.fft_plus.getDevicePtr(), matrix.wavefunction_plus.getDevicePtr(), CUFFT_INVERSE ), "iFFT Exec" );
    
    // Shift FFT Once again for visualization
    fft_shift_2D<<<grid_size, block_size>>>( matrix.fft_plus.getDevicePtr(), system.p.N_x, system.p.N_y );
    CHECK_CUDA_ERROR( {}, "FFT Shift" );
    
    // Do the same for the minus component
    if (not system.p.use_twin_mode)
        return;
    kernel_mask_fft<<<grid_size, block_size>>>( matrix.fft_minus.getDevicePtr(), matrix.fft_mask_minus.getDevicePtr(), system.p.N_x*system.p.N_y );
    CHECK_CUDA_ERROR( {}, "FFT Filter" )
    fft_shift_2D<<<grid_size, block_size>>>( matrix.fft_minus.getDevicePtr(), system.p.N_x,system.p.N_y );
    CHECK_CUDA_ERROR( {}, "FFT Shift" )
    CHECK_CUDA_ERROR( FFTSOLVER( plan, matrix.fft_minus.getDevicePtr(), matrix.wavefunction_minus.getDevicePtr(), CUFFT_INVERSE ), "iFFT Exec" );
    fft_shift_2D<<<grid_size, block_size>>>( matrix.fft_minus.getDevicePtr(), system.p.N_x,system.p.N_y );
    CHECK_CUDA_ERROR( {}, "FFT Shift" );
    #endif
}

/**
 * Split Step Fourier Method
 */
void PC3::Solver::iterateSplitStepFourier( dim3 block_size, dim3 grid_size ) {
    
    auto p = system.kernel_parameters;
    
    // This variable contains all the device pointers the kernel could need
    auto device_pointers = matrix.pointers();

    // Pointers to Oscillation Parameters
    auto pulse_pointers = dev_pulse_oscillation.pointers();
    auto pump_pointers = dev_pump_oscillation.pointers();
    auto potential_pointers = dev_potential_oscillation.pointers();

    // Liner Half Step
    // Calculate the FFT of Psi
    CHECK_CUDA_ERROR( FFTSOLVER( plan, (fft_complex_number*)device_pointers.wavefunction_plus, (fft_complex_number*)device_pointers.k1_wavefunction_plus, CUFFT_FORWARD ), "FFT Exec" );
    fft_shift_2D<<<grid_size, block_size>>>( device_pointers.k1_wavefunction_plus, system.p.N_x,system.p.N_y );
    
    CALL_KERNEL(
        Kernel::Compute::gp_scalar_linear_fourier, "linear_half_step", grid_size, block_size, 
        p.t, device_pointers, p, pulse_pointers, pump_pointers, potential_pointers,
        { 
            device_pointers.k1_wavefunction_plus, device_pointers.k1_wavefunction_minus, device_pointers.k1_reservoir_plus, device_pointers.k1_reservoir_minus,
            device_pointers.k2_wavefunction_plus, device_pointers.k2_wavefunction_minus, device_pointers.k2_reservoir_plus, device_pointers.k2_reservoir_minus
        }
    );
    fft_shift_2D<<<grid_size, block_size>>>( device_pointers.k2_wavefunction_plus, system.p.N_x,system.p.N_y );
    // Transform back.
    CHECK_CUDA_ERROR( FFTSOLVER( plan,  (fft_complex_number*)device_pointers.k2_wavefunction_plus,  (fft_complex_number*)device_pointers.k3_wavefunction_plus, CUFFT_INVERSE ), "iFFT Exec" );

    // Nonlinear Full Step
    CALL_KERNEL(
        Kernel::Compute::gp_scalar_nonlinear, "nonlinear_half_step", grid_size, block_size, 
        p.t, device_pointers, p, pulse_pointers, pump_pointers, potential_pointers,
        { 
            device_pointers.k2_wavefunction_plus, device_pointers.k2_wavefunction_minus, device_pointers.k2_reservoir_plus, device_pointers.k2_reservoir_minus,
            device_pointers.buffer_wavefunction_plus, device_pointers.buffer_wavefunction_minus, device_pointers.buffer_reservoir_plus, device_pointers.buffer_reservoir_minus
        }
    );

    // Liner Half Step
    // Calculate the FFT of Psi
    CHECK_CUDA_ERROR( FFTSOLVER( plan, (fft_complex_number*)device_pointers.buffer_wavefunction_plus, (fft_complex_number*)device_pointers.k4_wavefunction_plus, CUFFT_FORWARD ), "FFT Exec" );
    fft_shift_2D<<<grid_size, block_size>>>( device_pointers.k4_wavefunction_plus, system.p.N_x,system.p.N_y );
    CALL_KERNEL(
        Kernel::Compute::gp_scalar_linear_fourier, "linear_half_step", grid_size, block_size, 
        p.t, device_pointers, p, pulse_pointers, pump_pointers, potential_pointers,
        { 
            device_pointers.k4_wavefunction_plus, device_pointers.k4_wavefunction_minus, device_pointers.k4_reservoir_plus, device_pointers.k4_reservoir_minus,
            device_pointers.buffer_wavefunction_plus, device_pointers.buffer_wavefunction_minus, device_pointers.buffer_reservoir_plus, device_pointers.buffer_reservoir_minus
        }
    );
    fft_shift_2D<<<grid_size, block_size>>>( device_pointers.buffer_wavefunction_plus, system.p.N_x,system.p.N_y );
    // Transform back.
    CHECK_CUDA_ERROR( FFTSOLVER( plan,  (fft_complex_number*)device_pointers.buffer_wavefunction_plus,  (fft_complex_number*)device_pointers.wavefunction_plus, CUFFT_INVERSE ), "iFFT Exec" );

}

bool first_time = true;

/**
 * Iterates the Runge-Kutta-Method on the GPU
 * Note, that all device arrays and variables have to be initialized at this point
 * @param t Current time, will be updated to t + dt
 * @param dt Time step, will be updated to the next time step
 * @param N_x Number of grid points in one dimension
 * @param N_y Number of grid points in the other dimension
 */
bool PC3::Solver::iterate( ) {


    // First, check if the maximum time has been reached
    if ( system.p.t >= system.t_max )
        return false;

    dim3 block_size( system.block_size, 1 );
    dim3 grid_size( ( system.p.N_x*system.p.N_y + block_size.x ) / block_size.x, 1 );
    
    if (first_time and system.evaluateStochastic()) {
        first_time = false;
        auto device_pointers = matrix.pointers();
        CALL_KERNEL(
                PC3::Kernel::initialize_random_number_generator, "random_number_init", grid_size, block_size,
                system.random_seed, device_pointers.random_state, system.p.N_x*system.p.N_y
            );
        std::cout << "Initialized Random Number Generator" << std::endl;
    }
    
    if ( system.fixed_time_step )
        iterateFixedTimestepRungeKutta( block_size, grid_size );
    else
        iterateSplitStepFourier( block_size, grid_size );

    // Syncronize
    //CHECK_CUDA_ERROR( cudaDeviceSynchronize(), "Sync" );
    
    // Increase t.
    system.p.t = system.p.t + system.p.dt;

    // For statistical purposes, increase the iteration counter
    system.iteration++;

    // FFT Guard 
    if ( system.p.t - cached_t < system.fft_every )
        return true;

    // Calculate the FFT
    cached_t = system.p.t; 
    applyFFTFilter( block_size, grid_size, system.fft_mask.size() > 0 );

    return true;
    // Syncronize
    //CHECK_CUDA_ERROR( cudaDeviceSynchronize(), "Sync" );
}