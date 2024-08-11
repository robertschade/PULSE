#include <omp.h>

// Include Cuda Kernel headers
#include "cuda/typedef.cuh"
#include "kernel/kernel_compute.cuh"
#include "system/system_parameters.hpp"
#include "misc/helperfunctions.hpp"
#include "cuda/cuda_matrix.cuh"
#include "solver/gpu_solver.hpp"
#include "misc/commandline_io.hpp"


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

#define WAVEFUNCTION_PLUS_REAL 0
#define WAVEFUNCTION_PLUS_IMAG 1
#define RESERVOIR_PLUS_REAL 2
#define RESERVOIR_PLUS_IMAG 3
#define BUFFER_WAVEFUNCTION_PLUS_REAL 4
#define BUFFER_WAVEFUNCTION_PLUS_IMAG 5
#define BUFFER_RESERVOIR_PLUS_REAL 6
#define BUFFER_RESERVOIR_PLUS_IMAG 7
#define K1_WAVEFUNCTION_PLUS_REAL 8
#define K1_WAVEFUNCTION_PLUS_IMAG 9
#define K2_WAVEFUNCTION_PLUS_REAL 10
#define K2_WAVEFUNCTION_PLUS_IMAG 11
#define K3_WAVEFUNCTION_PLUS_REAL 12
#define K3_WAVEFUNCTION_PLUS_IMAG 13
#define K4_WAVEFUNCTION_PLUS_REAL 14
#define K4_WAVEFUNCTION_PLUS_IMAG 15
#define K1_RESERVOIR_PLUS_REAL 16
#define K1_RESERVOIR_PLUS_IMAG 17
#define K2_RESERVOIR_PLUS_REAL 18
#define K2_RESERVOIR_PLUS_IMAG 19
#define K3_RESERVOIR_PLUS_REAL 20
#define K3_RESERVOIR_PLUS_IMAG 21
#define K4_RESERVOIR_PLUS_REAL 22
#define K4_RESERVOIR_PLUS_IMAG 23

void PC3::Solver::iterateFixedTimestepRungeKutta4( dim3 block_size, dim3 grid_size ) {
    // This variable contains all the system parameters the kernel could need
    auto p = system.kernel_parameters;
    
    // This variable contains all the device pointers the kernel could need
    auto device_pointers = matrix.pointers();
    // Same IO every time
    Kernel::InputOutput io = { 
        device_pointers.wavefunction_plus, device_pointers.wavefunction_minus, 
        device_pointers.reservoir_plus, device_pointers.reservoir_minus, 
        device_pointers.buffer_wavefunction_plus, device_pointers.buffer_wavefunction_minus, 
        device_pointers.buffer_reservoir_plus, device_pointers.buffer_reservoir_minus 
    };

    // The CPU should briefly evaluate wether the stochastic kernel is used
    bool evaluate_stochastic = system.evaluateStochastic();

    // Pointers to Oscillation Parameters
    auto pulse_pointers = dev_pulse_oscillation.pointers();
    auto pump_pointers = dev_pump_oscillation.pointers();
    auto potential_pointers = dev_potential_oscillation.pointers();
    //transfer into local structure
    const int localblock_x=16;
    const int localblock_y=32;
    const int halo_x=4;
    const int halo_y=4;
    const int local_N_x=system.p.N_x/localblock_x;
    const int local_N_y=system.p.N_y/localblock_y;
    const int local_N_x_halo=local_N_x+2*halo_x;
    const int local_N_y_halo=local_N_y+2*halo_y;
    const int nobj=24;

    const bool dev=false;
    //const bool dev=true;

    if(!matrix.ld_init){
      int nth=omp_get_max_threads();
      std::cout << "N_x=" << system.p.N_x << " N_y=" << system.p.N_y << std::endl;
      std::cout << "local_N_x=" << local_N_x << " local_N_y=" << local_N_y << std::endl;
      std::cout << "local_N_x_halo=" << local_N_x_halo << " local_N_y_halo=" << local_N_y_halo << std::endl;
      std::cout << "number of threads=" << nth << std::endl;
      std::cout << "number of blocks=" << localblock_x*localblock_y << std::endl;
      std::cout << "number of objects=" << nobj << std::endl;
      std::cout << "local data size in bytes (one grid object)=" << sizeof(device_pointers.wavefunction_plus[0])*local_N_x_halo*local_N_y_halo << std::endl;
      std::cout << "local data size in bytes (all local data)=" << sizeof(device_pointers.wavefunction_plus[0])*local_N_x_halo*local_N_y_halo*nobj << std::endl;
      std::cout << "total local data size in bytes (one grid object)=" << sizeof(device_pointers.wavefunction_plus[0])*local_N_x_halo*local_N_y_halo*localblock_x*localblock_y << std::endl;
      std::cout << "total local data size in bytes (all local data)=" << sizeof(device_pointers.wavefunction_plus[0])*local_N_x_halo*local_N_y_halo*nobj*localblock_x*localblock_y << std::endl;
      //local data to improve cache locality, active size should fit into last level cache
      //for proper placement of the local data according to the NUMA first touch policy
      double t0=omp_get_wtime();
      if(not (nth<=localblock_x*localblock_y)){
        std::cout << "warning: number of blocks (localblock_x*localblock_y) should be larger or equal the number of threads" << std::endl;
      }
      if(system.p.N_x%localblock_x!=0){
        std::cout << "error: grid length in x is not divisible by localblock_x" << std::endl;
        exit(1);
      }
      if(system.p.N_y%localblock_y!=0){
        std::cout << "error: grid length in y is not divisible by localblock_y" << std::endl;
        exit(1);
      }
      if(local_N_x<halo_x or local_N_y<halo_y){
        std::cout << "error: the lendth/width of a local block must be at least the halo size" << std::endl;
        exit(1);
      }
      if(localblock_y*localblock_x%nth!=0){
        std::cout << "warning: localblock_x*localblock_y should be divisible by number of threads for an optimal distribution" << std::endl;
      }
      matrix.ld=new Type::real**[nobj];
      //matrix.ld=new Type::complex**[nobj];
      for(int i=0;i<nobj;i++){
        matrix.ld[i]=new Type::real*[localblock_x*localblock_y];
        //matrix.ld[i]=new Type::complex*[localblock_x*localblock_y];
      }
      #pragma omp parallel for schedule(static)
      for(int ib=0;ib<localblock_x*localblock_y;ib++){
        //std::cout << "thread " << omp_get_thread_num() << " " << ib << std::endl;
        for(int i=0;i<nobj;i++){
          matrix.ld[i][ib]=(Type::real*)aligned_alloc(64,local_N_x_halo*local_N_y_halo*sizeof(Type::real));
          //matrix.ld[i][ib]=(Type::complex*)aligned_alloc(64,local_N_x*local_N_y*sizeof(Type::complex));
          for(int ig=0;ig<local_N_x_halo*local_N_y_halo;ig++){
            matrix.ld[i][ib][ig]=0;
          }
        }
      }
      //build global-local map
      matrix.map=new int*[localblock_x*localblock_y];
      #pragma omp parallel for schedule(static)
      for(int ib=0;ib<localblock_x*localblock_y;ib++)
      {
        matrix.map[ib]=new int[local_N_x_halo*local_N_y_halo];
        int bx=ib%localblock_x;
        int by=(ib-bx)/localblock_y;
        //map point of local block with halo to global WITH HALO, the halo can then be used to apply boundary conditions
        for(int igy=0;igy<local_N_y_halo;igy++)
        {
          for(int igx=0;igx<local_N_x_halo;igx++)
          {
            //local id
            int ig=igy*local_N_x_halo+igx;
            //global id with global halo
            int ig2xg=(igx-halo_x+bx*local_N_x)+halo_x;
            int ig2yg=(igy-halo_y+by*local_N_y)+halo_y;
            if((ig2xg<halo_x or ig2xg>=system.p.N_x+halo_x) and system.p.periodic_boundary_x){
              ig2xg=(ig2xg-halo_x+system.p.N_x)%system.p.N_x+halo_x;
            }
            if((ig2yg<halo_y or ig2yg>=system.p.N_y+halo_y) and system.p.periodic_boundary_y){
              ig2yg=(ig2yg-halo_y+system.p.N_y)%system.p.N_y+halo_y;
            }
            //global id without global halo
            int ig2x=(ig2xg-halo_x);
            int ig2y=(ig2yg-halo_y);

            int ig2=ig2y*(system.p.N_x)+ig2x;
            //zero boundary conditions
            if((ig2xg<halo_x or ig2xg>=system.p.N_x+halo_x) and not system.p.periodic_boundary_x){
              ig2=-1;
            }
            if((ig2yg<halo_y or ig2yg>=system.p.N_y+halo_y) and not system.p.periodic_boundary_y){
              ig2=-1;
            }
            if(dev) std::cout << ib << " " << " " << bx << " " << by << " x="  << igx << " y=" << igy  << " x2=" << ig2x  << " y2=" <<ig2y  << " ig=" << ig  << " ig2=" << ig2 << std::endl;
            matrix.map[ib][ig]=ig2;
          }
        }
      }
      matrix.ld_init=true;
      //copy data to local structures
      #pragma omp parallel for schedule(static)
      for(int ib=0;ib<localblock_x*localblock_y;ib++)
      {
        for(int igy=0;igy<local_N_y_halo;igy++)
        {
          for(int igx=0;igx<local_N_x_halo;igx++)
          {
            int ig=igy*local_N_x_halo+igx;
            int ig2=matrix.map[ib][ig];
            if(ig2==-1){
              //zero boundary conditions
              matrix.ld[WAVEFUNCTION_PLUS_REAL][ib][ig]=0;
              matrix.ld[WAVEFUNCTION_PLUS_IMAG][ib][ig]=0;                   
              matrix.ld[RESERVOIR_PLUS_REAL][ib][ig]=0;
              matrix.ld[RESERVOIR_PLUS_IMAG][ib][ig]=0;
            }else{
              matrix.ld[WAVEFUNCTION_PLUS_REAL][ib][ig]=device_pointers.wavefunction_plus[ig2].real();
              matrix.ld[WAVEFUNCTION_PLUS_IMAG][ib][ig]=device_pointers.wavefunction_plus[ig2].imag();                   
              matrix.ld[RESERVOIR_PLUS_REAL][ib][ig]=device_pointers.reservoir_plus[ig2].real();
              matrix.ld[RESERVOIR_PLUS_IMAG][ib][ig]=device_pointers.reservoir_plus[ig2].imag();
            }
          }
        }
      }
      double t1=omp_get_wtime();
      std::cout << "tinit=" << t1-t0 << std::endl;
      std::cout << system.p.minus_i_over_h_bar_s << std::endl;
    }

    // PoC for cache-based copy kernel
    #pragma omp parallel for schedule(static)
    for(int ib=0;ib<localblock_x*localblock_y;ib++)
    {
      //compute step
      //CALCULATE_K( 1, p.t, wavefunction, reservoir );
      gp_scalar2(local_N_x_halo,local_N_y_halo,system.p.t,1,system.p.m2_over_dx2_p_dy2,system.p.one_over_dx2,system.p.one_over_dy2,system.p.minus_i_over_h_bar_s.imag(),system.p.g_c,system.p.g_r,system.p.R,system.p.gamma_c,system.p.gamma_r,system.p.m_eff_scaled,system.p.periodic_boundary_x,system.p.periodic_boundary_y,
          matrix.ld[WAVEFUNCTION_PLUS_REAL][ib],
          matrix.ld[WAVEFUNCTION_PLUS_IMAG][ib],
          matrix.ld[RESERVOIR_PLUS_REAL][ib],
          matrix.ld[RESERVOIR_PLUS_IMAG][ib],
          matrix.ld[K1_WAVEFUNCTION_PLUS_REAL][ib],
          matrix.ld[K1_WAVEFUNCTION_PLUS_IMAG][ib],
          matrix.ld[K1_RESERVOIR_PLUS_REAL][ib],
          matrix.ld[K1_RESERVOIR_PLUS_IMAG][ib]);
      
      //sum: 0.5*dt*K1  
      //    buffer_wavefunction_plus=wavefunction_plus+0.5*p.dt*k1_wavefunction_plus
      //    buffer_reservoir_plus=reservoir_plus+0.5*p.dt*k1_reservoir_plus
      sum_block_1(local_N_x_halo*local_N_y_halo,matrix.ld[WAVEFUNCTION_PLUS_REAL][ib],0.5*p.dt,matrix.ld[K1_WAVEFUNCTION_PLUS_REAL][ib],matrix.ld[BUFFER_WAVEFUNCTION_PLUS_REAL][ib]);
      sum_block_1(local_N_x_halo*local_N_y_halo,matrix.ld[WAVEFUNCTION_PLUS_IMAG][ib],0.5*p.dt,matrix.ld[K1_WAVEFUNCTION_PLUS_IMAG][ib],matrix.ld[BUFFER_WAVEFUNCTION_PLUS_IMAG][ib]);
      sum_block_1(local_N_x_halo*local_N_y_halo,matrix.ld[RESERVOIR_PLUS_REAL][ib],0.5*p.dt,matrix.ld[K1_RESERVOIR_PLUS_REAL][ib],matrix.ld[BUFFER_RESERVOIR_PLUS_REAL][ib]);
      sum_block_1(local_N_x_halo*local_N_y_halo,matrix.ld[RESERVOIR_PLUS_IMAG][ib],0.5*p.dt,matrix.ld[K1_RESERVOIR_PLUS_IMAG][ib],matrix.ld[BUFFER_RESERVOIR_PLUS_IMAG][ib]);
      
      //CALCULATE_K( 2, p.t + 0.5 * p.dt, buffer_wavefunction, buffer_reservoir );
      gp_scalar2(local_N_x_halo,local_N_y_halo,system.p.t+0.5*system.p.dt,2,system.p.m2_over_dx2_p_dy2,system.p.one_over_dx2,system.p.one_over_dy2,system.p.minus_i_over_h_bar_s.imag(),system.p.g_c,system.p.g_r,system.p.R,system.p.gamma_c,system.p.gamma_r,system.p.m_eff_scaled,system.p.periodic_boundary_x,system.p.periodic_boundary_y,
          matrix.ld[BUFFER_WAVEFUNCTION_PLUS_REAL][ib],
          matrix.ld[BUFFER_WAVEFUNCTION_PLUS_IMAG][ib],
          matrix.ld[BUFFER_RESERVOIR_PLUS_REAL][ib],
          matrix.ld[BUFFER_RESERVOIR_PLUS_IMAG][ib],
          matrix.ld[K2_WAVEFUNCTION_PLUS_REAL][ib],
          matrix.ld[K2_WAVEFUNCTION_PLUS_IMAG][ib],
          matrix.ld[K2_RESERVOIR_PLUS_REAL][ib],
          matrix.ld[K2_RESERVOIR_PLUS_IMAG][ib]);
      

      //sum: 0.5*dt*K2
      //    buffer_wavefunction_plus=wavefunction_plus+0.5*p.dt*k2_wavefunction_plus
      //    buffer_reservoir_plus=reservoir_plus+0.5*p.dt*k2_reservoir_plus
      sum_block_1(local_N_x_halo*local_N_y_halo,matrix.ld[WAVEFUNCTION_PLUS_REAL][ib],0.5*p.dt,matrix.ld[K2_WAVEFUNCTION_PLUS_REAL][ib],matrix.ld[BUFFER_WAVEFUNCTION_PLUS_REAL][ib]);
      sum_block_1(local_N_x_halo*local_N_y_halo,matrix.ld[WAVEFUNCTION_PLUS_IMAG][ib],0.5*p.dt,matrix.ld[K2_WAVEFUNCTION_PLUS_IMAG][ib],matrix.ld[BUFFER_WAVEFUNCTION_PLUS_IMAG][ib]);
      sum_block_1(local_N_x_halo*local_N_y_halo,matrix.ld[RESERVOIR_PLUS_REAL][ib],0.5*p.dt,matrix.ld[K2_RESERVOIR_PLUS_REAL][ib],matrix.ld[BUFFER_RESERVOIR_PLUS_REAL][ib]);
      sum_block_1(local_N_x_halo*local_N_y_halo,matrix.ld[RESERVOIR_PLUS_IMAG][ib],0.5*p.dt,matrix.ld[K2_RESERVOIR_PLUS_IMAG][ib],matrix.ld[BUFFER_RESERVOIR_PLUS_IMAG][ib]);
      
      //CALCULATE_K( 3, p.t + 0.5 * p.dt, buffer_wavefunction, buffer_reservoir);
      gp_scalar2(local_N_x_halo,local_N_y_halo,system.p.t+0.5*system.p.dt,3,system.p.m2_over_dx2_p_dy2,system.p.one_over_dx2,system.p.one_over_dy2,system.p.minus_i_over_h_bar_s.imag(),system.p.g_c,system.p.g_r,system.p.R,system.p.gamma_c,system.p.gamma_r,system.p.m_eff_scaled,system.p.periodic_boundary_x,system.p.periodic_boundary_y,
          matrix.ld[BUFFER_WAVEFUNCTION_PLUS_REAL][ib],
          matrix.ld[BUFFER_WAVEFUNCTION_PLUS_IMAG][ib],
          matrix.ld[BUFFER_RESERVOIR_PLUS_REAL][ib],
          matrix.ld[BUFFER_RESERVOIR_PLUS_IMAG][ib],
          matrix.ld[K3_WAVEFUNCTION_PLUS_REAL][ib],
          matrix.ld[K3_WAVEFUNCTION_PLUS_IMAG][ib],
          matrix.ld[K3_RESERVOIR_PLUS_REAL][ib],
          matrix.ld[K3_RESERVOIR_PLUS_IMAG][ib]);
      
      
      //sum: dt*K3
      //    buffer_wavefunction_plus=wavefunction_plus+p.dt*k3_wavefunction_plus
      //    buffer_reservoir_plus=reservoir_plus+p.dt*k3_reservoir_plus
      sum_block_1(local_N_x_halo*local_N_y_halo,matrix.ld[WAVEFUNCTION_PLUS_REAL][ib],p.dt,matrix.ld[K3_WAVEFUNCTION_PLUS_REAL][ib],matrix.ld[BUFFER_WAVEFUNCTION_PLUS_REAL][ib]);
      sum_block_1(local_N_x_halo*local_N_y_halo,matrix.ld[WAVEFUNCTION_PLUS_IMAG][ib],p.dt,matrix.ld[K3_WAVEFUNCTION_PLUS_IMAG][ib],matrix.ld[BUFFER_WAVEFUNCTION_PLUS_IMAG][ib]);
      sum_block_1(local_N_x_halo*local_N_y_halo,matrix.ld[RESERVOIR_PLUS_REAL][ib],p.dt,matrix.ld[K3_RESERVOIR_PLUS_REAL][ib],matrix.ld[BUFFER_RESERVOIR_PLUS_REAL][ib]);
      sum_block_1(local_N_x_halo*local_N_y_halo,matrix.ld[RESERVOIR_PLUS_IMAG][ib],p.dt,matrix.ld[K3_RESERVOIR_PLUS_IMAG][ib],matrix.ld[BUFFER_RESERVOIR_PLUS_IMAG][ib]);
        
      //CALCULATE_K( 4, p.t + p.dt, buffer_wavefunction, buffer_reservoir);
      gp_scalar2(local_N_x_halo,local_N_y_halo,system.p.t+system.p.dt,4,system.p.m2_over_dx2_p_dy2,system.p.one_over_dx2,system.p.one_over_dy2,system.p.minus_i_over_h_bar_s.imag(),system.p.g_c,system.p.g_r,system.p.R,system.p.gamma_c,system.p.gamma_r,system.p.m_eff_scaled,system.p.periodic_boundary_x,system.p.periodic_boundary_y,
          matrix.ld[BUFFER_WAVEFUNCTION_PLUS_REAL][ib],
          matrix.ld[BUFFER_WAVEFUNCTION_PLUS_IMAG][ib],
          matrix.ld[BUFFER_RESERVOIR_PLUS_REAL][ib],
          matrix.ld[BUFFER_RESERVOIR_PLUS_IMAG][ib],
          matrix.ld[K4_WAVEFUNCTION_PLUS_REAL][ib],
          matrix.ld[K4_WAVEFUNCTION_PLUS_IMAG][ib],
          matrix.ld[K4_RESERVOIR_PLUS_REAL][ib],
          matrix.ld[K4_RESERVOIR_PLUS_IMAG][ib]);
      
      //sum: { 1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0 } // RK Final Weights
      //    buffer_wavefunction_plus=wavefunction_plus+p.dt/6*k1_wavefunction_plus+p.dt/3*k2_wavefunction_plus+p.dt/3*k3_wavefunction_plus+p.dt/6*k4_wavefunction_plus
      //    buffer_reservoir_plus=reservoir_plus+p.dt/6*k1_reservoir_plus+p.dt/3*k2_reservoir_plus+p.dt/3*k3_reservoir_plus+p.dt/6*k4_reservoir_plus
      
      sum_block_4(local_N_x_halo*local_N_y_halo,matrix.ld[WAVEFUNCTION_PLUS_REAL][ib], 
          p.dt/6.0,matrix.ld[K1_WAVEFUNCTION_PLUS_REAL][ib], 
          p.dt/3.0,matrix.ld[K2_WAVEFUNCTION_PLUS_REAL][ib], 
          p.dt/3.0,matrix.ld[K3_WAVEFUNCTION_PLUS_REAL][ib], 
          p.dt/6.0,matrix.ld[K4_WAVEFUNCTION_PLUS_REAL][ib], 
          matrix.ld[BUFFER_WAVEFUNCTION_PLUS_REAL][ib]);
      sum_block_4(local_N_x_halo*local_N_y_halo,matrix.ld[WAVEFUNCTION_PLUS_IMAG][ib], 
          p.dt/6.0,matrix.ld[K1_WAVEFUNCTION_PLUS_IMAG][ib], 
          p.dt/3.0,matrix.ld[K2_WAVEFUNCTION_PLUS_IMAG][ib], 
          p.dt/3.0,matrix.ld[K3_WAVEFUNCTION_PLUS_IMAG][ib], 
          p.dt/6.0,matrix.ld[K4_WAVEFUNCTION_PLUS_IMAG][ib], 
          matrix.ld[BUFFER_WAVEFUNCTION_PLUS_IMAG][ib]);
      sum_block_4(local_N_x_halo*local_N_y_halo,matrix.ld[RESERVOIR_PLUS_REAL][ib], 
          p.dt/6.0,matrix.ld[K1_RESERVOIR_PLUS_REAL][ib], 
          p.dt/3.0,matrix.ld[K2_RESERVOIR_PLUS_REAL][ib], 
          p.dt/3.0,matrix.ld[K3_RESERVOIR_PLUS_REAL][ib], 
          p.dt/6.0,matrix.ld[K4_RESERVOIR_PLUS_REAL][ib], 
          matrix.ld[BUFFER_RESERVOIR_PLUS_REAL][ib]);
      sum_block_4(local_N_x_halo*local_N_y_halo,matrix.ld[RESERVOIR_PLUS_IMAG][ib], 
          p.dt/6.0,matrix.ld[K1_RESERVOIR_PLUS_IMAG][ib], 
          p.dt/3.0,matrix.ld[K2_RESERVOIR_PLUS_IMAG][ib], 
          p.dt/3.0,matrix.ld[K3_RESERVOIR_PLUS_IMAG][ib], 
          p.dt/6.0,matrix.ld[K4_RESERVOIR_PLUS_IMAG][ib], 
          matrix.ld[BUFFER_RESERVOIR_PLUS_IMAG][ib]);
      
      /*
      //copy data to global structures
      for(int igy=halo_y;igy<local_N_y+halo_y;igy++)
      {
        for(int igx=halo_x;igx<local_N_x+halo_x;igx++)
        {
          int ig=igy*local_N_x_halo+igx;
          int ig2=matrix.map[ib][ig];
//          if(ig2!=-1){
            device_pointers.wavefunction_plus[ig2]=(matrix.ld[BUFFER_WAVEFUNCTION_PLUS_REAL][ib][ig],matrix.ld[BUFFER_WAVEFUNCTION_PLUS_IMAG][ib][ig]);
            device_pointers.reservoir_plus[ig2]=(matrix.ld[BUFFER_RESERVOIR_PLUS_REAL][ib][ig],matrix.ld[BUFFER_RESERVOIR_PLUS_IMAG][ib][ig]);
//          }
        }
      }*/
    }

    if(dev){
      //compute iteration with former code

      CALCULATE_K( 1, p.t, wavefunction, reservoir );
   
      CALL_KERNEL(
          Kernel::RK::runge_sum_to_input_kw, "Sum for K4", grid_size, block_size,
          p.dt, device_pointers, p, io,
          { 0.5 } // 0.5*dt*K1
      );

      CALCULATE_K( 2, p.t + 0.5 * p.dt, buffer_wavefunction, buffer_reservoir );
      
      CALL_KERNEL(
          Kernel::RK::runge_sum_to_input_kw, "Sum for K3", grid_size, block_size,
          p.dt, device_pointers, p, io,
          {0.0, 0.5 } // 0.5*dt*K2
      );

      CALCULATE_K( 3, p.t + 0.5 * p.dt, buffer_wavefunction, buffer_reservoir);

      CALL_KERNEL(
          Kernel::RK::runge_sum_to_input_kw, "Sum for K4", grid_size, block_size,
          p.dt, device_pointers, p, io,
          { 0.0, 0.0, 1.0 } // dt*K3
      );

      CALCULATE_K( 4, p.t + p.dt, buffer_wavefunction, buffer_reservoir);

      CALL_KERNEL(
          Kernel::RK::runge_sum_to_input_kw, "Final Sum", grid_size, block_size,
          p.dt, device_pointers, p, io,
          { 1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0 } // RK Final Weights
      );

      //compare results
      double d=0;
      for(int ib=0;ib<localblock_x*localblock_y;ib++)
      {
        for(int igy=halo_y;igy<local_N_y+halo_y;igy++)
        {
          for(int igx=halo_x;igx<local_N_x+halo_x;igx++)
          {
            int ig=igy*local_N_x_halo+igx;
            int ig2=matrix.map[ib][ig];

            double d0=abs(device_pointers.buffer_wavefunction_plus[ig2].real()-matrix.ld[BUFFER_WAVEFUNCTION_PLUS_REAL][ib][ig])+abs(device_pointers.buffer_wavefunction_plus[ig2].imag()-matrix.ld[BUFFER_WAVEFUNCTION_PLUS_IMAG][ib][ig]);
            if(d0>1e-10)std::cout << ib << " " << igy << " " << igx << " " << device_pointers.buffer_wavefunction_plus[ig2] << " " <<  matrix.ld[BUFFER_WAVEFUNCTION_PLUS_REAL][ib][ig] << " " << matrix.ld[BUFFER_WAVEFUNCTION_PLUS_IMAG][ib][ig] << " " << d0 << std::endl;
//            double d0=abs(device_pointers.wavefunction_plus[ig2].real()-matrix.ld[WAVEFUNCTION_PLUS_REAL][ib][ig])+abs(device_pointers.wavefunction_plus[ig2].imag()-matrix.ld[WAVEFUNCTION_PLUS_IMAG][ib][ig]);
//            if(d0>1e-10)std::cout << ib << " " << igy << " " << igx << " " << device_pointers.wavefunction_plus[ig2] << " " <<  matrix.ld[WAVEFUNCTION_PLUS_REAL][ib][ig] << " " << matrix.ld[WAVEFUNCTION_PLUS_IMAG][ib][ig] << " " << d0 << std::endl;
//            double d0=abs(device_pointers.buffer_reservoir_plus[ig2].real()-matrix.ld[BUFFER_RESERVOIR_PLUS_REAL][ib][ig])+abs(device_pointers.buffer_reservoir_plus[ig2].imag()-matrix.ld[BUFFER_RESERVOIR_PLUS_IMAG][ib][ig]);
//            if(d0>1e-10)std::cout << ib << " " << igy << " " << igx << " " << device_pointers.buffer_reservoir_plus[ig2] << " " <<  matrix.ld[BUFFER_RESERVOIR_PLUS_REAL][ib][ig] << " " << matrix.ld[BUFFER_RESERVOIR_PLUS_IMAG][ib][ig] << " " << d0 << std::endl;
//            double d0=abs(device_pointers.k1_wavefunction_plus[ig2].real()-matrix.ld[K1_WAVEFUNCTION_PLUS_REAL][ib][ig])+abs(device_pointers.k1_wavefunction_plus[ig2].imag()-matrix.ld[K1_WAVEFUNCTION_PLUS_IMAG][ib][ig]);
//            if(d0>1e-10)std::cout << ib << " " << igy << " " << igx << " " << device_pointers.k1_wavefunction_plus[ig2] << " " <<  matrix.ld[K1_WAVEFUNCTION_PLUS_REAL][ib][ig] << " " << matrix.ld[K1_WAVEFUNCTION_PLUS_IMAG][ib][ig] << " " << d0 << std::endl;
//            double d0=abs(device_pointers.k2_wavefunction_plus[ig2].real()-matrix.ld[K2_WAVEFUNCTION_PLUS_REAL][ib][ig])+abs(device_pointers.k2_wavefunction_plus[ig2].imag()-matrix.ld[K2_WAVEFUNCTION_PLUS_IMAG][ib][ig]);
//            if(d0>1e-10)std::cout << ib << " " << igy << " " << igx << " " << device_pointers.k2_wavefunction_plus[ig2] << " " <<  matrix.ld[K2_WAVEFUNCTION_PLUS_REAL][ib][ig] << " " << matrix.ld[K2_WAVEFUNCTION_PLUS_IMAG][ib][ig] << " " << d0 << std::endl;
//            double d0=abs(device_pointers.k3_wavefunction_plus[ig2].real()-matrix.ld[K3_WAVEFUNCTION_PLUS_REAL][ib][ig])+abs(device_pointers.k3_wavefunction_plus[ig2].imag()-matrix.ld[K3_WAVEFUNCTION_PLUS_IMAG][ib][ig]);
//            if(d0>1e-10)std::cout << ib << " " << igy << " " << igx << " " << device_pointers.k3_wavefunction_plus[ig2] << " " <<  matrix.ld[K3_WAVEFUNCTION_PLUS_REAL][ib][ig] << " " << matrix.ld[K3_WAVEFUNCTION_PLUS_IMAG][ib][ig] << " " << d0 << std::endl;
//            double d0=abs(device_pointers.k4_wavefunction_plus[ig2].real()-matrix.ld[K4_WAVEFUNCTION_PLUS_REAL][ib][ig])+abs(device_pointers.k4_wavefunction_plus[ig2].imag()-matrix.ld[K4_WAVEFUNCTION_PLUS_IMAG][ib][ig]);
//            if(d0>1e-10)std::cout << ib << " " << igy << " " << igx << " " << device_pointers.k4_wavefunction_plus[ig2] << " " <<  matrix.ld[K4_WAVEFUNCTION_PLUS_REAL][ib][ig] << " " << matrix.ld[K4_WAVEFUNCTION_PLUS_IMAG][ib][ig] << " " << d0 << std::endl;
            d+=d0;
          }
        }
      }
      std::cout << "diff=" << d << std::endl;
      exit(1);
    }
    // Swap the next and current wavefunction buffers. This only swaps the pointers, not the data.
    swapBuffers();
    
    return;
}

inline void PC3::Solver::sum_block_1(int N,Type::real* __restrict A,Type::real alpha,Type::real* __restrict B,Type::real* __restrict C){
  #pragma omp simd
  for(int i=0;i<N;i++)
  {
    C[i]=A[i]+alpha*B[i];
  } 
}
inline void PC3::Solver::sum_block_4(int N,Type::real* __restrict A,Type::real alpha1,Type::real* __restrict B1,Type::real alpha2,Type::real* __restrict B2,Type::real alpha3,Type::real* __restrict B3,Type::real alpha4,Type::real* __restrict B4,Type::real* __restrict C){
  #pragma omp simd
  for(int i=0;i<N;i++)
  {
    C[i]=A[i]+alpha1*B1[i]+alpha2*B2[i]+alpha3*B3[i]+alpha4*B4[i];
  } 
}
void PC3::Solver::gp_scalar2(int Nx,int Ny,Type::real t,int shrink,Type::real m2_over_dx2_p_dy2,Type::real one_over_dx2,Type::real one_over_dy2,Type::real minus_1_over_h_bar_s,Type::real g_c,Type::real g_r,Type::real R,Type::real gamma_c,Type::real gamma_r,Type::real m_eff_scaled,bool periodic_boundary_x,bool periodic_boundary_y,Type::real* __restrict in_wfu_real,Type::real* __restrict in_wfu_imag,Type::real* __restrict in_res_real,Type::real* __restrict in_res_imag,Type::real* __restrict out_wfu_real,Type::real* __restrict out_wfu_imag,Type::real* __restrict out_res_real,Type::real* __restrict out_res_imag){
  for(int igy=shrink;igy<Ny-shrink;igy++)
  {
    for(int igx=shrink;igx<Nx-shrink;igx++)
    {
      int ig=igy*Nx+igx;
      Type::real hamilton_real = m2_over_dx2_p_dy2 * in_wfu_real[ig];
      Type::real hamilton_imag = m2_over_dx2_p_dy2 * in_wfu_imag[ig];

      hamilton_real+=one_over_dx2*(in_wfu_real[(igy)*Nx+(igx-1)]+in_wfu_real[(igy)*Nx+(igx+1)])
        +one_over_dy2*(in_wfu_real[(igy-1)*Nx+(igx)]+in_wfu_real[(igy+1)*Nx+(igx)]);
      hamilton_imag+=one_over_dx2*(in_wfu_imag[(igy)*Nx+(igx-1)]+in_wfu_imag[(igy)*Nx+(igx+1)])
        +one_over_dy2*(in_wfu_imag[(igy-1)*Nx+(igx)]+in_wfu_imag[(igy+1)*Nx+(igx)]);

      Type::real in_psi_norm=in_wfu_real[ig]*in_wfu_real[ig]+in_wfu_imag[ig]*in_wfu_imag[ig];
      Type::real result_real=-minus_1_over_h_bar_s*m_eff_scaled*hamilton_imag;
      Type::real result_imag=minus_1_over_h_bar_s*m_eff_scaled*hamilton_real;

//          result += p_in.minus_i_over_h_bar_s * p_in.g_c * in_psi_norm * in_wf;
      result_real+=-minus_1_over_h_bar_s*g_c*in_psi_norm*in_wfu_imag[ig];
      result_imag+= minus_1_over_h_bar_s*g_c*in_psi_norm*in_wfu_real[ig];
//          result += p_in.minus_i_over_h_bar_s * p_in.g_r * in_rv * in_wf
      result_real+=-minus_1_over_h_bar_s * g_r*(in_wfu_real[ig]*in_res_imag[ig]+in_wfu_imag[ig]*in_res_real[ig]);
      result_imag+= minus_1_over_h_bar_s * g_r*(in_wfu_real[ig]*in_res_real[ig]-in_wfu_imag[ig]*in_res_imag[ig]);
//          result += Type::real(0.5) * p_in.R * in_rv * in_wf;
      result_real+=Type::real(0.5) * R*(in_wfu_real[ig]*in_res_real[ig]-in_wfu_imag[ig]*in_res_imag[ig]);
      result_real+=Type::real(0.5) * R*(in_wfu_real[ig]*in_res_imag[ig]+in_wfu_imag[ig]*in_res_real[ig]);
//          //result += Type::real(0.5) * p.R * in_rv * in_wf;
//          result -= Type::real(0.5) * p_in.gamma_c * in_wf;
      result_real-=Type::real(0.5) * gamma_c*in_wfu_real[ig];
      result_imag-=Type::real(0.5) * gamma_c*in_wfu_imag[ig];
      
      out_wfu_real[ig]=result_real;
      out_wfu_imag[ig]=result_imag;

//          result = -p_in.gamma_r * in_rv;
      Type::real result2_real=-gamma_r*in_res_real[ig];
      Type::real result2_imag=-gamma_r*in_res_imag[ig];
//          result -= p_in.R * in_psi_norm * in_rv;
      result2_real-=R*in_psi_norm*in_res_real[ig];
      result2_imag-=R*in_psi_norm*in_res_imag[ig];
      
      out_res_real[ig]=result2_real;
      out_res_imag[ig]=result2_imag;
    }
  }
}
