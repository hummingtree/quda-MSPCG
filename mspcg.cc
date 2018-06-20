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
    mat(NULL), MdagM(NULL), mat_precondition(NULL), MdagM_precondition(NULL),
    r(NULL), p(NULL), z(NULL), mmp(NULL), tmp(NULL), fr(NULL), fz(NULL), 
    immp(NULL), ip(NULL), itmp(NULL),
    dr(NULL), dp(NULL), dmmp(NULL), dtmp(NULL)
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
    MdagM = new DiracMdagM(mat);
     
    mat_sloppy = new DiracMobiusPC(dirac_param_sloppy);
    MdagM_sloppy = new DiracMdagM(mat_sloppy);
    
    mat_precondition = new DiracMobiusPC(dirac_param_precondition);
    MdagM_precondition = new DiracMdagM(mat_precondition);

    fillInnerSolverParam(solver_prec_param, param);

    printfQuda("MSPCG constructor ends.\n");

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
    delete MdagM_precondition;
    delete mat_precondition;
    
    delete MdagM_sloppy;
    delete mat_sloppy;
    
    delete MdagM;
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
    (*MdagM)(*tx, *tb, *tt);
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
    (*MdagM_precondition)(*fx, *fb, *ft);
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
      (*MdagM_precondition)(*immp, *ip, *itmp);
      profile.TPSTART(QUDA_PROFILE_FREE);
      zero_extended_color_spinor_interface( *immp, R, QUDA_CUDA_FIELD_LOCATION, 0);
      profile.TPSTOP(QUDA_PROFILE_FREE);
      //zero_boundary_fermion(eg, mmp);
      Mpk2 = reDotProduct(*ip, *immp);

      alpha = rk2 / Mpk2; 

      axpy(alpha, *ip, ix);
      axpy(-alpha, *immp, ib);

      rkp12 = blas::norm2(ib);

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
    (*MdagM)(*dr, dx, *dtmp); // r = MdagM * x
    double rk2 = xmyNorm(db, *dr); // r = b - MdagM * x
    
    printfQuda("outer_cg: before starting: r2 = %8.4e \n", rk2);
    blas::copy(*dp, *dr);

    for(int loop_count = 0; loop_count < param.maxiter; loop_count++){
      (*MdagM)(*dmmp, *dp, *dtmp);
      Mpk2 = reDotProduct(*dp, *dmmp);

      alpha = rk2 / Mpk2; 

      axpy(alpha, *dp, dx);
      axpy(-alpha, *dmmp, *dr);

      rkp12 = blas::norm2(*dr);
      if(rkp12 < quit) break;
      
      beta = rkp12 / rk2;
      rk2 = rkp12;

      xpay(ib, beta, *dp);

      printfQuda("outer_cg: #%04d: r2 = %8.4e alpha = %8.4e beta = %8.4e Mpk2 = %8.4e\n", loop_count, rk2, alpha, beta, Mpk2);
    }

    return loop_count;
  }

  void MSPCG::operator()(ColorSpinorField& dx, ColorSpinorField& db)
  {

//    test_dslash( b ); 
    int parity = 0;
    Gflops = 0.;
    fGflops = 0.;

    profile.TPSTART(QUDA_PROFILE_PREAMBLE);
    // Check to see that we're not trying to invert on a zero-field source
    double b2 = norm2(b);
    if(b2 == 0.){
      profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
      printfQuda("Warning: inverting on zero-field source\n");
      x = b;
      param.true_res = 0.;
      param.true_res_hq = 0.;
    }

    // initializing the fermion vectors.
    ColorSpinorParam csParam(b);
    csParam.create = QUDA_ZERO_FIELD_CREATE;

// d* means precise
    dr  =  new cudaColorSpinorField(csParam);
    dp  =  new cudaColorSpinorField(csParam);
    dmmp = new cudaColorSpinorField(csParam);
    dtmp = new cudaColorSpinorField(csParam);

// sloppy
    csParam.setPrecision(dirac_param_sloppy.gauge->Precision());
    
    r  =   new cudaColorSpinorField(csParam);
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
    //    int parity = MdagM->getMatPCType();

    double alpha, beta, rkzk, pkApk, zkP1rkp1, zkrk;

    double stop = stopping(param.tol, b2, param.residual_type);

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);

    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    // end of initializing.

    for(int cycle=0; cycle<3; cycle++){
      
      (*MdagM)(*dr, dx, *dtmp); // r = MdagM * x
      double r2 = xmyNorm(db, *dr); // r = b - MdagM * x
      printfQuda("True precise residual is %8.4e\n", r2);
    
      double sloppy_solver_stop = r2*param.tol*param.tol>stop ? r2*param.tol*param.tol : stop;

      blas::copy(*r, *dr); // throw true residual into the sloppy solver.
     
      copyExtendedColorSpinor(*fr, *r, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
      inner_cg(*fz, *fr);
      copyExtendedColorSpinor(*z, *fz, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
      blas::copy(*p, *z);

      k = 0;
      while( k < param.maxiter ){
        rkzk = reDotProduct(*r, *z);

        (*MdagM)(*mmp, *p, *tmp);
        pkApk = reDotProduct(*p, *mmp);
        alpha = rkzk / pkApk;

        axpy(alpha, *p, x); // x_k+1 = x_k + alpha * p_k
        axpy(-alpha, *mmp, *r); // r_k+1 = r_k - alpha * Ap_k
        double rr2 = blas::norm2(*r);
        if(rr2 < sloppy_solver_stop) break;

        // z = M^-1 r
        profile.TPSTART(QUDA_PROFILE_FREE);
        copyExtendedColorSpinor(*fr, *r, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
        profile.TPSTOP(QUDA_PROFILE_FREE);
        inner_cg(*fz, *fr);
        profile.TPSTART(QUDA_PROFILE_FREE);
        copyExtendedColorSpinor(*z, *fz, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
        profile.TPSTOP(QUDA_PROFILE_FREE);

        zkP1rkp1 = reDotProduct(*z, *r);
        beta = zkP1rkp1 / rkzk;
        xpay(*z, beta, *p);

        double zz2 = blas::norm2(*z);
        printfQuda("z2/r2: %8.4e/%8.4e.\n", zz2, rr2);

        ++k;
        printfQuda("MSPCG/iter.count/r2/target_r2/%%/target_%%: %05d %8.4e %8.4e %8.4e %8.4e\n", k, rr2, stop, std::sqrt(rr2/b2), param.tol);

      }

      blas::copy(*dtmp, *x);
      xpy(*dtmp, dx);

    }
    
    k = outer_cg(dx, db, stop);

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);

    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    Gflops = MdagM->flops()*1e-9+blas::flops*1e-9;
    fGflops = MdagM_precondition->flops()*1e-9+blas::flops*1e-9;
    param.gflops = Gflops;
    param.iter += k;

    reduceDouble(Gflops);
    reduceDouble(fGflops);

    if (k==param.maxiter) warningQuda("Exceeded maximum iterations %d", param.maxiter);

    // compute the true residual 

    (*MdagM)(*dr, dx, *tmp);
    double true_res = xmyNorm(db, *dr);
    param.true_res = sqrt(true_res/b2);

    printfQuda("True residual/target_r2: %8.4e/%8.4e.\n", true_res, stop);
    printfQuda("Performance outer: %8.4f TFLOPS.\n", Gflops*1e-3/param.secs);
    printfQuda("Performance inter: %8.4f TFLOPS.\n", fGflops*1e-3/param.secs);
    printfQuda("Performance total: %8.4f TFLOPS.\n", (Gflops+fGflops)*1e-3/param.secs);

    // reset the flops counters
    blas::flops = 0;

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);

    delete r;
    delete z;
    delete p;
    delete mmp;
    delete tmp;
    
    delete fr;
    delete fz;
    
    delete immp;
    delete ip;
    delete itmp;

    delete dr;
    delete dp;
    delete dmmp;
    delete dtmp;

    return;
  }


} // namespace quda
