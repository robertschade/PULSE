#include "kernel/kernel_compute.cuh"
#include "kernel/kernel_hamilton.cuh"
#include "kernel/kernel_index_overwrite.cuh"

/**
 * Mode without TE/TM Splitting
 * The differential equation for this model reduces to
 * ...
 */
PULSE_GLOBAL void PC3::Kernel::Compute::gp_scalar( int i, Type::real t, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p_in, Solver::TemporalEvelope::Pointers oscillation_pulse, Solver::TemporalEvelope::Pointers oscillation_pump, Solver::TemporalEvelope::Pointers oscillation_potential, InputOutput io ) {
    
    LOCAL_SHARE_STRUCT( SystemParameters::KernelParameters, p_in, p );

    OVERWRITE_THREAD_INDEX( i );

    const Type::complex in_wf = io.in_wf_plus[i];
    const Type::complex in_rv = io.in_rv_plus[i];

    Type::complex hamilton = p_in.m2_over_dx2_p_dy2 * in_wf;
    hamilton += PC3::Kernel::Hamilton::scalar_neighbours( io.in_wf_plus, i, i / p_in.N_x /*Row*/, i % p_in.N_x /*Col*/, p_in.N_x, p_in.N_y, p_in.one_over_dx2, p_in.one_over_dy2, p_in.periodic_boundary_x, p_in.periodic_boundary_y );

    const Type::real in_psi_norm = CUDA::abs2( in_wf );
    
    // MARK: Wavefunction
    Type::complex result = p_in.minus_i_over_h_bar_s * ( p_in.m_eff_scaled * hamilton );

    for (int k = 0; k < oscillation_potential.n; k++) {
        const size_t offset = k * p_in.N_x * p_in.N_y;
        const Type::complex potential = dev_ptrs.potential_plus[i+offset] * oscillation_potential.amp[k];//CUDA::gaussian_oscillator(t, oscillation_potential.t0[k], oscillation_potential.sigma[k], oscillation_potential.freq[k]);
        result += p_in.minus_i_over_h_bar_s * potential * in_wf;
    }

    result += p_in.minus_i_over_h_bar_s * p_in.g_c * in_psi_norm * in_wf;
    result += (p_in.minus_i_over_h_bar_s * p_in.g_r +Type::real(0.5) * p_in.R) * in_rv * in_wf;
//    result += Type::real(0.5) * p.R * in_rv * in_wf;
    result -= Type::real(0.5) * p_in.gamma_c * in_wf;


    // MARK: Pulse
    for (int k = 0; k < oscillation_pulse.n; k++) {
        const size_t offset = k * p_in.N_x * p_in.N_y;
        const Type::complex pulse = dev_ptrs.pulse_plus[i+offset];
        result += p_in.one_over_h_bar_s * pulse * oscillation_pulse.amp[k];//CUDA::gaussian_complex_oscillator(t, oscillation_pulse.t0[k], oscillation_pulse.sigma[k], oscillation_pulse.freq[k]);
    }
    
    // MARK: Stochastic
    if (p_in.stochastic_amplitude > 0.0) {
        const Type::complex dw = dev_ptrs.random_number[i] * CUDA::sqrt( ( p_in.R * in_rv + p_in.gamma_c ) / (Type::real(4.0) * p_in.dV) );
        result -= p_in.minus_i_over_h_bar_s * p_in.g_c * in_wf / p_in.dV - dw / p_in.dt;
    }
    
    io.out_wf_plus[i] = result;
    
    // MARK: Reservoir
    result = -p_in.gamma_r * in_rv;
    result -= p_in.R * in_psi_norm * in_rv;
    for (int k = 0; k < oscillation_pump.n; k++) {
        const int offset = k * p_in.N_x * p_in.N_y;
            result += dev_ptrs.pump_plus[i+offset] * oscillation_pump.amp[k];
    }

    // MARK: Stochastic-2
    if (p_in.stochastic_amplitude > 0.0)
        result += p_in.R * in_rv / p_in.dV;

    io.out_rv_plus[i] = result;

}

/**
 * Linear, Nonlinear and Independet parts of the upper Kernel
 * These isolated implementations serve for the Split Step
 * Fourier Method (SSFM)
*/

PULSE_GLOBAL void PC3::Kernel::Compute::gp_scalar_linear_fourier( int i, Type::real t, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p, Solver::TemporalEvelope::Pointers oscillation_pulse, Solver::TemporalEvelope::Pointers oscillation_pump, Solver::TemporalEvelope::Pointers oscillation_potential, InputOutput io ) {
    
    OVERWRITE_THREAD_INDEX( i );

    // We do some weird looking casting to avoid intermediate casts to size_t
    Type::real row = Type::real(size_t(i / p.N_x));
    Type::real col = Type::real(size_t(i % p.N_x));
    
    const Type::real k_x = 2.0*3.1415926535 * Type::real(col <= p.N_x/2 ? col : -Type::real(p.N_x) + col)/p.L_x;
    const Type::real k_y = 2.0*3.1415926535 * Type::real(row <= p.N_y/2 ? row : -Type::real(p.N_y) + row)/p.L_y;

    Type::real linear = p.h_bar_s/2.0/p.m_eff * (k_x*k_x + k_y*k_y);
    io.out_wf_plus[i] = io.in_wf_plus[i] / Type::real(p.N2) * CUDA::exp( p.minus_i * linear * p.dt / Type::real(2.0) );
}

PULSE_GLOBAL void PC3::Kernel::Compute::gp_scalar_nonlinear( int i, Type::real t, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p, Solver::TemporalEvelope::Pointers oscillation_pulse, Solver::TemporalEvelope::Pointers oscillation_pump, Solver::TemporalEvelope::Pointers oscillation_potential, InputOutput io ) {
    
    OVERWRITE_THREAD_INDEX( i );
    
    const Type::complex in_wf = io.in_wf_plus[i];
    const Type::complex in_rv = io.in_rv_plus[i];
    const Type::real in_psi_norm = CUDA::abs2( in_wf );
    
    // MARK: Wavefunction
    Type::complex result = {p.g_c * in_psi_norm, -p.h_bar_s * Type::real(0.5) * p.gamma_c};

    for (int k = 0; k < oscillation_potential.n; k++) {
        const size_t offset = k * p.N_x * p.N_y;
        const Type::complex potential = dev_ptrs.potential_plus[i+offset] * oscillation_potential.amp[k]; //CUDA::gaussian_oscillator(t, oscillation_potential.t0[k], oscillation_potential.sigma[k], oscillation_potential.freq[k]);
        result += potential;
    }

    result += p.g_r * in_rv;
    result += p.i * p.h_bar_s * Type::real(0.5) * p.R * in_rv;

    // MARK: Stochastic
    if (p.stochastic_amplitude > 0.0) {
        const Type::complex dw = dev_ptrs.random_number[i] * CUDA::sqrt( ( p.R * in_rv + p.gamma_c ) / (Type::real(4.0) * p.dV) );
        result -= p.g_c / p.dV;
    }

    io.out_wf_plus[i] = in_wf * CUDA::exp(p.minus_i_over_h_bar_s * result * p.dt);

    // MARK: Reservoir
    result = -p.gamma_r * in_rv;
    result -= p.R * in_psi_norm * in_rv;
    for (int k = 0; k < oscillation_pump.n; k++) {
        const int offset = k * p.N_x * p.N_y;
        result += dev_ptrs.pump_plus[i+offset] * oscillation_pump.amp[k]; //CUDA::gaussian_oscillator(t, oscillation_pump.t0[k], oscillation_pump.sigma[k], oscillation_pump.freq[k]);
    }
    // MARK: Stochastic-2
    if (p.stochastic_amplitude > 0.0)
        result += p.R * in_rv / p.dV;
    io.out_rv_plus[i] = in_rv + result * p.dt;
}

PULSE_GLOBAL void PC3::Kernel::Compute::gp_scalar_independent( int i, Type::real t, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p, Solver::TemporalEvelope::Pointers oscillation_pulse, Solver::TemporalEvelope::Pointers oscillation_pump, Solver::TemporalEvelope::Pointers oscillation_potential, InputOutput io ) {
    
    OVERWRITE_THREAD_INDEX( i );
    Type::complex result = {0.0,0.0};

    // MARK: Pulse
    for (int k = 0; k < oscillation_pulse.n; k++) {
        const size_t offset = k * p.N_x * p.N_y;
        const Type::complex pulse = dev_ptrs.pulse_plus[i+offset];
        result += p.minus_i_over_h_bar_s * p.dt * pulse * oscillation_pulse.amp[k]; //CUDA::gaussian_complex_oscillator(t, oscillation_pulse.t0[k], oscillation_pulse.sigma[k], oscillation_pulse.freq[k]);
    }
    if (p.stochastic_amplitude > 0.0) {
        const Type::complex in_rv = io.in_rv_plus[i];
        const Type::complex dw = dev_ptrs.random_number[i] * CUDA::sqrt( ( p.R * in_rv + p.gamma_c ) / (Type::real(4.0) * p.dV) );
        result += dw;
    }
    io.out_wf_plus[i] = io.in_wf_plus[i] + result;
}
