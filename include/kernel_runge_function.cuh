#pragma once
#include "cuda_complex_math.cuh"
#include "cuda_device_variables.cuh"
#include "kernel_hamilton.cuh"

CUDA_GLOBAL void rungeFuncKernel( int i, real_number t, complex_number* __restrict__ in_Psi_Plus, complex_number* __restrict__ in_Psi_Minus, complex_number* __restrict__ in_n_Plus, complex_number* __restrict__ in_n_Minus, complex_number* __restrict__ k_Psi_Plus, complex_number* __restrict__ k_Psi_Minus, complex_number* __restrict__ k_n_Plus, complex_number* __restrict__ k_n_Minus,
                                 /* Pump Parameters */ real_number* __restrict__ dev_pump_cache_Plus, real_number* __restrict__ dev_pump_cache_Minus,
                                 /* Pulse Parameters */ real_number* __restrict__ dev_pulse_t0, real_number* __restrict__ dev_pulse_amp, real_number* __restrict__ dev_pulse_freq, real_number* __restrict__ dev_pulse_sigma, int* __restrict__ dev_pulse_m, int* __restrict__ dev_pulse_pol, real_number* __restrict__ dev_pulse_width, real_number* __restrict__ dev_pulse_X, real_number* __restrict__ dev_pulse_Y, bool evaluate_pulse );