#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>

#include <test_util.h>

#include <mspcg.h>

#include <gauge_tools.h>

extern quda::cudaGaugeField* gaugePrecondition;
extern quda::cudaGaugeField* gaugePrecise;
extern quda::cudaGaugeField* gaugeSloppy;


namespace quda {

  using namespace blas;

  static cudaGaugeField* createExtendedGauge(cudaGaugeField &in, const int *R, TimeProfile &profile,
      bool redundant_comms=false, QudaReconstructType recon=QUDA_RECONSTRUCT_INVALID)
  {
    int y[4];
    for (int dir=0; dir<4; ++dir) y[dir] = in.X()[dir] + 2*R[dir];
    int pad = 0;

    GaugeFieldParam gParamEx(y, in.Precision(), recon != QUDA_RECONSTRUCT_INVALID ? recon : in.Reconstruct(), pad,
        in.Geometry(), QUDA_GHOST_EXCHANGE_EXTENDED);
    gParamEx.create = QUDA_ZERO_FIELD_CREATE;
    gParamEx.order = in.Order();
    gParamEx.siteSubset = QUDA_FULL_SITE_SUBSET;
    gParamEx.t_boundary = in.TBoundary();
    gParamEx.nFace = 1;
    gParamEx.tadpole = in.Tadpole();
    for (int d=0; d<4; d++) gParamEx.r[d] = R[d];

    cudaGaugeField *out = new cudaGaugeField(gParamEx);

    // copy input field into the extended device gauge field
    copyExtendedGauge(*out, in, QUDA_CUDA_FIELD_LOCATION);

    // now fill up the halos
    profile.TPSTART(QUDA_PROFILE_COMMS);
    out->exchangeExtendedGhost(R,redundant_comms);
    profile.TPSTOP(QUDA_PROFILE_COMMS);

    return out;
  } 


  // set the required parameters for the inner solver
  static void fillInnerSolverParam(SolverParam& inner, const SolverParam& outer)
  {
    //    inner.tol = outer.tol_precondition;
    inner.tol = 5e-2;
    inner.maxiter = outer.maxiter_precondition;
    inner.delta = 1e-20; // no reliable updates within the inner solver
    inner.precision = outer.precision_precondition; // preconditioners are uni-precision solvers
    inner.precision_sloppy = outer.precision_precondition;

    inner.iter = 0;
    inner.gflops = 0;
    inner.secs = 0;

    inner.inv_type_precondition = QUDA_INVALID_INVERTER;
    inner.is_preconditioner = true; // used to tell the inner solver it is an inner solver

    inner.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  }

  MSPCG::MSPCG(QudaInvertParam* inv_param, SolverParam& _param, TimeProfile& profile, int ps) :
    Solver(_param, profile), solver_prec(0), solver_prec_param(_param), 
    mat(NULL), nrm_op(NULL), mat_precondition(NULL), nrm_op_precondition(NULL), mat_sloppy(NULL), nrm_op_sloppy(NULL),
    r(NULL), p(NULL), z(NULL), mmp(NULL), tmp(NULL), fr(NULL), fz(NULL), 
    immp(NULL), ip(NULL), itmp(NULL),
    vct_dr(NULL), vct_dp(NULL), vct_dmmp(NULL), vct_dtmp(NULL)
  { 

    printfQuda("MSPCG constructor starts.\n");

    R[0]=1;
    R[1]=2;
    R[2]=2;
    R[3]=2;
    // TODO: R is the checkerboarded size.


    if(inv_param->dslash_type != QUDA_MOBIUS_DWF_DSLASH){
      errorQuda("ONLY works for QUDA_MOBIUS_DWF_DSLASH.");
    }

    // create extended gauge field
    // TODO: dynamical allocation need fix
    if(not gaugePrecondition){
      errorQuda("gaugePrecondition not valid.");
    }

    int gR[4] = {2*R[0], R[1], R[2], R[3]}; 
//    bool p2p = comm_peer2peer_enabled_global();
//    comm_enable_peer2peer(false); // The following function does NOT work with peer2peer comm
    padded_gauge_field = createExtendedGauge(*gaugePrecise, gR, profile, true, QUDA_RECONSTRUCT_NO);
    padded_gauge_field_precondition = createExtendedGauge(*gaugePrecondition, gR, profile, true, QUDA_RECONSTRUCT_NO);
//    comm_enable_peer2peer(p2p);

//    printfQuda( "Original gauge field = %16.12e\n", plaquette( *gaugePrecise, QUDA_CUDA_FIELD_LOCATION ).x );
//    printfQuda( "Extended gauge field = %16.12e\n", plaquette( *padded_gauge_field, QUDA_CUDA_FIELD_LOCATION ).x );

    //DiracParam dirac_param;
    setDiracParam(dirac_param, inv_param, true); // pc = true

    setDiracParam(dirac_param_sloppy, inv_param, true); // pc = true
    dirac_param_sloppy.gauge = gaugeSloppy;

    setDiracParam(dirac_param_precondition, inv_param, true); // pc = true
    dirac_param_precondition.gauge = padded_gauge_field_precondition;

    for(int i = 0; i < 4; i++){
      dirac_param.commDim[i] = 1; 
      dirac_param_sloppy.commDim[i] = 1; 
      dirac_param_precondition.commDim[i] = 0;
    }

    dirac_param.print();
    dirac_param_sloppy.print();
    dirac_param_precondition.print();
    
    mat = new DiracMobiusPC(dirac_param);
    nrm_op = new DiracMdagM(mat);
     
    mat_sloppy = new DiracMobiusPC(dirac_param_sloppy);
    nrm_op_sloppy = new DiracMdagM(mat_sloppy);
    
    mat_precondition = new DiracMobiusPC(dirac_param_precondition);
    nrm_op_precondition = new DiracMdagM(mat_precondition);

    fillInnerSolverParam(solver_prec_param, param);

    printfQuda("MSPCG constructor ends.\n");
    
    copier_timer.Reset("woo", "hoo", 0);
    precise_timer.Reset("woo", "hoo", 0);
    sloppy_timer.Reset("woo", "hoo", 0);
    preconditioner_timer.Reset("woo", "hoo", 0);
  }

  MSPCG::~MSPCG(){
    profile.TPSTART(QUDA_PROFILE_FREE);
    /*
       if(solver_prec) 
       delete solver_prec;

       if( MdagM ) 
       delete MdagM;
       if( MdagM_precondition ) 
       delete MdagM_precondition;
       if( mat )
       delete mat;
       if( mat_precondition )
       delete mat_precondition;
       */
    delete nrm_op_precondition;
    delete mat_precondition;
    
    delete nrm_op_sloppy;
    delete mat_sloppy;
    
    delete nrm_op;
    delete mat;

    delete padded_gauge_field;
    delete padded_gauge_field_precondition;

    profile.TPSTOP(QUDA_PROFILE_FREE);
  }

  void MSPCG::test_dslash( const ColorSpinorField& b ){
    
    ColorSpinorParam csParam(b);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    csParam.print();
    // TODO: def
    cudaColorSpinorField* tx = NULL;
    cudaColorSpinorField* tt = NULL;
    cudaColorSpinorField* tb = NULL;

    tx  = new cudaColorSpinorField(csParam);
    tt  = new cudaColorSpinorField(csParam);
    tb  = new cudaColorSpinorField(csParam);

    blas::copy( *tb, b );

    double b2 = blas::norm2(*tb);
    printfQuda("Test b2 before = %16.12e.\n", b2);
    if( comm_rank() ){ blas::zero(*tb); }
    b2 = blas::norm2(*tb);
    printfQuda("Test b2 after  = %16.12e.\n", b2);
    (*nrm_op)(*tx, *tb, *tt);
    double x2 = blas::norm2(*tx);
    printfQuda("Test     x2/b2 = %16.12e/%16.12e.\n", x2, b2);
    if( comm_rank() ){
      blas::zero(*tx);
    }
    x2 = blas::norm2(*tx);
    printfQuda("Chopping x2/b2 = %16.12e/%16.12e.\n", x2, b2);
    
    cudaColorSpinorField* fx = NULL;
    cudaColorSpinorField* fb = NULL;
    cudaColorSpinorField* ft = NULL;

    for(int i=0; i<4; ++i){
      csParam.x[i] += 2*R[i];
    }

    csParam.setPrecision(dirac_param_precondition.gauge->Precision());
    csParam.print();

    // TODO: def
    fx  = new cudaColorSpinorField(csParam);
    fb  = new cudaColorSpinorField(csParam);
    ft  = new cudaColorSpinorField(csParam);
    blas::zero(*fb);
    blas::zero(*fx);

    copyExtendedColorSpinor(*fb, *tb, QUDA_CUDA_FIELD_LOCATION, 0, NULL, NULL, NULL, NULL); // parity = 0

    //    quda::pack::initConstants(*dirac_param_precondition.gauge, profile);
    double fb2 = norm2(*fb);
    (*nrm_op_precondition)(*fx, *fb, *ft);
    double fx2 = norm2(*fx);
    printfQuda("Test   fx2/fb2 = %16.12e/%16.12e.\n", fx2, fb2);
    zero_extended_color_spinor_interface( *fx, R, QUDA_CUDA_FIELD_LOCATION, 0);
    fx2 = norm2(*fx);
    printfQuda("Chopping   fx2 = %16.12e.\n", fx2);

    copyExtendedColorSpinor(*tx, *fx, QUDA_CUDA_FIELD_LOCATION, 0, NULL, NULL, NULL, NULL); // parity = 0
    double x2_ = blas::norm2(*tx);
    printfQuda("Rebuild     x2 = %16.12e.\n", x2_);
    printfQuda("%% diff      x2 = %16.12e (This number is SUPPOSED to be tiny).\n", (x2-x2_)/x2);

    delete tx;
    delete tt;
    delete tb;
    delete fx;
    delete fb;
    delete ft;

    printfQuda("dslash test completed.\n");
  }

  void MSPCG::inner_cg(ColorSpinorField& ix, ColorSpinorField& ib )
  {
    commGlobalReductionSet(false);

    blas::zero(ix);

    double rk2 = blas::norm2(ib);
    double Mpk2, MdagMpk2, alpha, beta, rkp12;

    printfQuda("inner_cg: before starting: r2 = %8.4e \n", rk2);
    blas::copy(*ip, ib);

    for(int local_loop_count = 0; local_loop_count < 5; local_loop_count++){
      preconditioner_timer.Start("woo", "hoo", 0);
      (*nrm_op_precondition)(*immp, *ip, *itmp);
      preconditioner_timer.Stop("woo", "hoo", 0);
      
      zero_extended_color_spinor_interface( *immp, R, QUDA_CUDA_FIELD_LOCATION, 0);
      //zero_boundary_fermion(eg, mmp);
      copier_timer.Start("woo", "hoo", 0);
      Mpk2 = reDotProduct(*ip, *immp);
      copier_timer.Stop("woo", "hoo", 0);

      alpha = rk2 / Mpk2; 

      axpy(alpha, *ip, ix);
      rkp12 = axpyNorm(-alpha, *immp, ib);

      
      beta = rkp12 / rk2;
      rk2 = rkp12;

      xpay(ib, beta, *ip);
      
      printfQuda("inner_cg: #%04d: r2 = %8.4e alpha = %8.4e beta = %8.4e Mpk2 = %8.4e\n",
          local_loop_count, rk2, alpha, beta, Mpk2);
    }

    commGlobalReductionSet(true);

    return;
  }

  int MSPCG::outer_cg( ColorSpinorField& dx, ColorSpinorField& db, double quit )
  {
    double Mpk2, MdagMpk2, alpha, beta, rkp12;
    precise_timer.Start("woo", "hoo", 0);
    (*nrm_op)(*vct_dr, dx, *vct_dtmp); // r = nrm_op * x
    precise_timer.Stop("woo", "hoo", 0);
    double rk2 = xmyNorm(db, *vct_dr); // r = b - nrm_op * x
    
    printfQuda("outer_cg: before starting: r2 = %8.4e \n", rk2);
    if(rkp12 < quit){
      printfQuda("outer_cg: CONVERGED with ZERO effort.\n");
      return 0;
    }
    
    blas::copy(*vct_dp, *vct_dr);
    
    int loop_count;
    for(loop_count = 0; loop_count < param.maxiter; loop_count++){
      
      precise_timer.Start("woo", "hoo", 0);
      (*nrm_op)(*vct_dmmp, *vct_dp, *vct_dtmp);
      precise_timer.Stop("woo", "hoo", 0);
      Mpk2 = reDotProduct(*vct_dp, *vct_dmmp);

      alpha = rk2 / Mpk2; 

      axpy(alpha, *vct_dp, dx);
      rkp12 = axpyNorm(-alpha, *vct_dmmp, *vct_dr);

//      rkp12 = blas::norm2(*vct_dr);
      
      beta = rkp12 / rk2;
      rk2 = rkp12;
      if(rkp12 < quit) break;

      xpay(*vct_dr, beta, *vct_dp);

      printfQuda("outer_cg: #%04d: r2 = %8.4e alpha = %8.4e beta = %8.4e Mpk2 = %8.4e\n", loop_count, rk2, alpha, beta, Mpk2);
    }
    
    printfQuda("outer_cg: CONVERGED after %04d iterations: r2/target_r2 = %8.4e/%8.4e.\n", loop_count+1, rk2, quit);

    return loop_count;
  }

  void MSPCG::operator()(ColorSpinorField& dx, ColorSpinorField& db)
  {

//    test_dslash( db ); 
    int parity = 0;
    Gflops = 0.;
    fGflops = 0.;
    
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);
    // Check to see that we're not trying to invert on a zero-field source
    double b2 = norm2(db);
    if(b2 == 0.){
      profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
      printfQuda("Warning: inverting on zero-field source\n");
      dx = db;
      param.true_res = 0.;
      param.true_res_hq = 0.;
    }

    // initializing the fermion vectors.
    ColorSpinorParam csParam(db);
    csParam.create = QUDA_ZERO_FIELD_CREATE;

// d* means precise
    vct_dr  =  new cudaColorSpinorField(csParam);
    vct_dp  =  new cudaColorSpinorField(csParam);
    vct_dmmp = new cudaColorSpinorField(csParam);
    vct_dtmp = new cudaColorSpinorField(csParam);

// sloppy
    csParam.setPrecision(dirac_param_sloppy.gauge->Precision());
    
    r  =   new cudaColorSpinorField(csParam);
    x  =   new cudaColorSpinorField(csParam);
    z  =   new cudaColorSpinorField(csParam);
    p  =   new cudaColorSpinorField(csParam);
    mmp  = new cudaColorSpinorField(csParam);
    tmp  = new cudaColorSpinorField(csParam);
    
    for(int i=0; i<4; ++i){
      csParam.x[i] += 2*R[i];
    }
    csParam.setPrecision(dirac_param_precondition.gauge->Precision());

// f* means fine/preconditioning
    fr  =  new cudaColorSpinorField(csParam);
    fz  =  new cudaColorSpinorField(csParam);

// i* means inner preconditioning
    immp=  new cudaColorSpinorField(csParam);
    ip  =  new cudaColorSpinorField(csParam);
    itmp=  new cudaColorSpinorField(csParam);

    blas::zero(*fr);

    int k;
    //    int parity = nrm_op->getMatPCType();
    double alpha, beta, rkzk, pkApk, zkP1rkp1, zkrk;

    double stop = stopping(param.tol, b2, param.residual_type);

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);

    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    // end of initializing.

    for(int cycle=0; cycle<5; cycle++){
      
      precise_timer.Start("woo", "hoo", 0);
      (*nrm_op)(*vct_dr, dx, *vct_dtmp); // r = MdagM * x
      precise_timer.Stop("woo", "hoo", 0);
      double r2 = xmyNorm(db, *vct_dr); // r = b - MdagM * x
      printfQuda("Cycle #%02d.\n", cycle);
      printfQuda("True precise residual is %8.4e\n", r2);
      if(r2 < stop) break;
    
      double sloppy_solver_stop = r2*param.tol*param.tol*1e4>stop ? r2*param.tol*param.tol*1e4 : stop;

      blas::copy(*r, *vct_dr); // throw true residual into the sloppy solver.
      blas::zero(*x);
      copier_timer.Start("woo", "hoo", 0);
      copyExtendedColorSpinor(*fr, *r, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
      copier_timer.Stop("woo", "hoo", 0);
      inner_cg(*fz, *fr);
      copier_timer.Start("woo", "hoo", 0);
      copyExtendedColorSpinor(*z, *fz, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
      copier_timer.Stop("woo", "hoo", 0);
      blas::copy(*p, *z);

      k = 0;
      while( k < param.maxiter ){
        rkzk = reDotProduct(*r, *z);

        sloppy_timer.Start("woo", "hoo", 0);
        (*nrm_op_sloppy)(*mmp, *p, *tmp);
        sloppy_timer.Stop("woo", "hoo", 0);
        pkApk = reDotProduct(*p, *mmp);
        alpha = rkzk / pkApk;

        axpy(alpha, *p, *x); // x_k+1 = x_k + alpha * p_k
        double rr2 = axpyNorm(-alpha, *mmp, *r); // r_k+1 = r_k - alpha * Ap_k
//        double rr2 = blas::norm2(*r);
        if(rr2 < sloppy_solver_stop) break;

        // z = M^-1 r
        copier_timer.Start("woo", "hoo", 0);
        copyExtendedColorSpinor(*fr, *r, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
        copier_timer.Stop("woo", "hoo", 0);
        
        inner_cg(*fz, *fr);
        
        copier_timer.Start("woo", "hoo", 0);
        copyExtendedColorSpinor(*z, *fz, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
        copier_timer.Stop("woo", "hoo", 0);

        zkP1rkp1 = reDotProduct(*z, *r);
        beta = zkP1rkp1 / rkzk;
        xpay(*z, beta, *p);

        double zz2 = blas::norm2(*z);
        printfQuda("z2/r2: %8.4e/%8.4e.\n", zz2, rr2);

        ++k;
        printfQuda("MSPCG/iter.count/r2/target_r2/%%/target_%%: %05d %8.4e %8.4e %8.4e %8.4e\n", k, rr2, stop, std::sqrt(rr2/b2), param.tol);

      }

      blas::copy(*vct_dtmp, *x);
      xpy(*vct_dtmp, dx);

    }
    
    k = outer_cg(dx, db, stop);

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);

    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    param.iter += k;

    double precise_tflops = nrm_op->flops()*1e-12;
    double sloppy_tflops = nrm_op_sloppy->flops()*1e-12;
    double preconditioner_tflops = nrm_op_precondition->flops()*1e-12;
    reduceDouble(precise_tflops);
    reduceDouble(sloppy_tflops);
    reduceDouble(preconditioner_tflops);
    param.gflops = sloppy_tflops;

    double prec_time = preconditioner_timer.time; 
    reduceMaxDouble(prec_time);

    if (k==param.maxiter) warningQuda("Exceeded maximum iterations %d", param.maxiter);

    // compute the true residual 

    (*nrm_op)(*vct_dr, dx, *vct_dtmp);
    double true_res = xmyNorm(db, *vct_dr);
    param.true_res = sqrt(true_res/b2);

    printfQuda("True residual/target_r2: %8.4e/%8.4e.\n", true_res, stop);
    printfQuda("Performance precise:        %8.4f TFLOPS in %8.4f secs with %05d calls.\n", 
      precise_tflops/precise_timer.time, precise_timer.time, precise_timer.count);
    printfQuda("Performance sloppy:         %8.4f TFLOPS in %8.4f secs with %05d calls.\n", 
      sloppy_tflops/sloppy_timer.time, sloppy_timer.time, sloppy_timer.count);
    printfQuda("Performance preconditioner: %8.4f TFLOPS in %8.4f secs with %05d calls.\n", 
      preconditioner_tflops/prec_time, prec_time, preconditioner_timer.count);
    printfQuda("Performance copier:                         in %8.4f secs with %05d calls.\n",
      copier_timer.time, copier_timer.count);

    // reset the flops counters
    blas::flops = 0;

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);

    profile.TPSTART(QUDA_PROFILE_FREE);
    delete r;
    delete x;
    delete z;
    delete p;
    delete mmp;
    delete tmp;
    
    delete fr;
    delete fz;
    
    delete immp;
    delete ip;
    delete itmp;

    delete vct_dr;
    delete vct_dp;
    delete vct_dmmp;
    delete vct_dtmp;
    profile.TPSTOP(QUDA_PROFILE_FREE);

    return;
  }


} // namespace quda
