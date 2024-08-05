#include "cuda/typedef.cuh"
#include "kernel/kernel_compute.cuh"
#include "kernel/kernel_index_overwrite.cuh"

// Summs one K
PULSE_GLOBAL void PC3::Kernel::RK::runge_sum_to_input_ki( int i, Type::real dt, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p, InputOutput io ) {
    OVERWRITE_THREAD_INDEX(i);

    io.out_wf_plus[i] = dev_ptrs.wavefunction_plus[i] + dt * io.in_wf_plus[i];
    io.out_rv_plus[i] = dev_ptrs.reservoir_plus[i] + dt * io.in_rv_plus[i];
    if ( not p.use_twin_mode ) 
        return;
    io.out_wf_minus[i] = dev_ptrs.wavefunction_minus[i] + dt * io.in_wf_minus[i];
    io.out_rv_minus[i] = dev_ptrs.reservoir_minus[i] + dt * io.in_rv_minus[i];
}

// Sums all Ks with weights. Oh yes, this looks terrible. For this to be pretty, we would need to create 
// yet another struct that holds all the buffers in an array. OR: we do the smart thing and restructure
// the original dev_ptrs struct to hold all the buffers in an array. This would make the code much more
// readable and maintainable. TODO
PULSE_GLOBAL void PC3::Kernel::RK::runge_sum_to_input_kw( int i, Type::real dt, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p, InputOutput io, const RK::Weights weights ){
    OVERWRITE_THREAD_INDEX(i);
    Type::complex wf = 0.0;
    Type::complex rv = 0.0;
    for (int n = weights.start; n < weights.n; n++) {
        const auto w = weights.weights[n];
        if(w==0.0)continue;
        switch (n) { 
            case 0: wf += w * dev_ptrs.k1_wavefunction_plus[i]; rv += w * dev_ptrs.k1_reservoir_plus[i]; break;
            case 1: wf += w * dev_ptrs.k2_wavefunction_plus[i]; rv += w * dev_ptrs.k2_reservoir_plus[i]; break;
            case 2: wf += w * dev_ptrs.k3_wavefunction_plus[i]; rv += w * dev_ptrs.k3_reservoir_plus[i]; break;
            case 3: wf += w * dev_ptrs.k4_wavefunction_plus[i]; rv += w * dev_ptrs.k4_reservoir_plus[i]; break;
            case 4: wf += w * dev_ptrs.k5_wavefunction_plus[i]; rv += w * dev_ptrs.k5_reservoir_plus[i]; break;
            case 5: wf += w * dev_ptrs.k6_wavefunction_plus[i]; rv += w * dev_ptrs.k6_reservoir_plus[i]; break;
            case 6: wf += w * dev_ptrs.k7_wavefunction_plus[i]; rv += w * dev_ptrs.k7_reservoir_plus[i]; break;
            case 7: wf += w * dev_ptrs.k8_wavefunction_plus[i]; rv += w * dev_ptrs.k8_reservoir_plus[i]; break;
            case 8: wf += w * dev_ptrs.k9_wavefunction_plus[i]; rv += w * dev_ptrs.k9_reservoir_plus[i]; break;
            case 9: wf += w * dev_ptrs.k10_wavefunction_plus[i]; rv += w * dev_ptrs.k10_reservoir_plus[i]; break;
        }
    }
   
    io.out_wf_plus[i] = io.in_wf_plus[i] + dt * wf;
    io.out_rv_plus[i] = io.in_rv_plus[i] + dt * rv;
    if ( not p.use_twin_mode ) 
        return;
   
    Type::complex wf2 = 0.0;
    Type::complex rv2 = 0.0;
    for (int n = weights.start; n < weights.n; n++) {
        const auto w = weights.weights[n];
        if(w==0.0)continue;
        switch (n) {
            case 0: wf2 += w * dev_ptrs.k1_wavefunction_minus[i]; rv2 += w * dev_ptrs.k1_reservoir_minus[i]; break;
            case 1: wf2 += w * dev_ptrs.k2_wavefunction_minus[i]; rv2 += w * dev_ptrs.k2_reservoir_minus[i]; break;
            case 2: wf2 += w * dev_ptrs.k3_wavefunction_minus[i]; rv2 += w * dev_ptrs.k3_reservoir_minus[i]; break;
            case 3: wf2 += w * dev_ptrs.k4_wavefunction_minus[i]; rv2 += w * dev_ptrs.k4_reservoir_minus[i]; break;
            case 4: wf2 += w * dev_ptrs.k5_wavefunction_minus[i]; rv2 += w * dev_ptrs.k5_reservoir_minus[i]; break;
            case 5: wf2 += w * dev_ptrs.k6_wavefunction_minus[i]; rv2 += w * dev_ptrs.k6_reservoir_minus[i]; break;
            case 6: wf2 += w * dev_ptrs.k7_wavefunction_minus[i]; rv2 += w * dev_ptrs.k7_reservoir_minus[i]; break;
            case 7: wf2 += w * dev_ptrs.k8_wavefunction_minus[i]; rv2 += w * dev_ptrs.k8_reservoir_minus[i]; break;
            case 8: wf2 += w * dev_ptrs.k9_wavefunction_minus[i]; rv2 += w * dev_ptrs.k9_reservoir_minus[i]; break;
            case 9: wf2 += w * dev_ptrs.k10_wavefunction_minus[i]; rv2 += w * dev_ptrs.k10_reservoir_minus[i]; break;
        }
    }
    
    io.out_wf_minus[i] = io.in_wf_minus[i] + dt * wf2;
    io.out_rv_minus[i] = io.in_rv_minus[i] + dt * rv2;
  
}


PULSE_GLOBAL void PC3::Kernel::RK::runge_sum_to_error( int i, Type::real dt, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p, RK::Weights weights ) {
    OVERWRITE_THREAD_INDEX(i);
    // The first weigth is for the input wavefunction, the rest are for the Ks
    Type::complex wf = weights.weights[0] * dev_ptrs.buffer_wavefunction_plus[i];
    for (int n = 1; n < weights.n; n++) {
        const auto w = weights.weights[n];
        switch (n) { 
            case 1: wf += w * dev_ptrs.k1_wavefunction_plus[i]; break;
            case 2: wf += w * dev_ptrs.k2_wavefunction_plus[i]; break;
            case 3: wf += w * dev_ptrs.k3_wavefunction_plus[i]; break;
            case 4: wf += w * dev_ptrs.k4_wavefunction_plus[i]; break;
            case 5: wf += w * dev_ptrs.k5_wavefunction_plus[i]; break;
            case 6: wf += w * dev_ptrs.k6_wavefunction_plus[i]; break;
            case 7: wf += w * dev_ptrs.k7_wavefunction_plus[i]; break;
            case 8: wf += w * dev_ptrs.k8_wavefunction_plus[i]; break;
            case 9: wf += w * dev_ptrs.k9_wavefunction_plus[i]; break;
            case 10: wf += w * dev_ptrs.k10_wavefunction_plus[i]; break;
        }
    }
    
    dev_ptrs.rk_error[i] = CUDA::abs2(dt * wf);
    if ( not p.use_twin_mode ) 
        return;
    
    wf = weights.weights[0] * dev_ptrs.buffer_wavefunction_minus[i];
    for (int n = 1; n < weights.n; n++) {
        const auto w = weights.weights[n];
        switch (n) {
            case 1: wf += w * dev_ptrs.k1_wavefunction_minus[i]; break;
            case 2: wf += w * dev_ptrs.k2_wavefunction_minus[i]; break;
            case 3: wf += w * dev_ptrs.k3_wavefunction_minus[i]; break;
            case 4: wf += w * dev_ptrs.k4_wavefunction_minus[i]; break;
            case 5: wf += w * dev_ptrs.k5_wavefunction_minus[i]; break;
            case 6: wf += w * dev_ptrs.k6_wavefunction_minus[i]; break;
            case 7: wf += w * dev_ptrs.k7_wavefunction_minus[i]; break;
            case 8: wf += w * dev_ptrs.k8_wavefunction_minus[i]; break;
            case 9: wf += w * dev_ptrs.k9_wavefunction_minus[i]; break;
            case 10: wf += w * dev_ptrs.k10_wavefunction_minus[i]; break;
        }
    }
    
    dev_ptrs.rk_error[i] += p.i*CUDA::abs2(dt * wf);
}
