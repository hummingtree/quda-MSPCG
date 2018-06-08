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

#include <mspcg.h>

extern quda::cudaGaugeField* gaugePrecondition;
extern quda::cudaGaugeField* gaugePrecise;

namespace quda {

  using namespace blas;
	
	static cudaGaugeField* createExtendedGauge(cudaGaugeField &in, const int *R, TimeProfile &profile,
						   bool redundant_comms=false, QudaReconstructType recon=QUDA_RECONSTRUCT_INVALID)
	{
	  profile.TPSTART(QUDA_PROFILE_INIT);
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
	
	  profile.TPSTOP(QUDA_PROFILE_INIT);
	
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
    Solver(_param, profile), solver_prec(0), solver_prec_param(_param), mat(NULL), MdagM(NULL)
  { // note that for consistency will accept ONLY M and WILL promote that to MdagM.
  
		// setVerbosity(QUDA_DEBUG_VERBOSE);

		printfQuda("MSPCG constructor starts.\n");
		
		R.fill(ps);

		if(inv_param->dslash_type != QUDA_MOBIUS_DWF_DSLASH){
			errorQuda("ONLY works for QUDA_MOBIUS_DWF_DSLASH.");
		}

		// create extended gauge field
		// TODO: dynamical allocation need fix
		if(not gaugePrecondition){
			errorQuda("gaugePrecondition not valid.");
		}

		int gR[4] = {2*R[0], R[1], R[2], R[3]}; 
		cudaGaugeField* padded_gauge_field_precondition = createExtendedGauge(*gaugePrecondition, gR, profile, true, QUDA_RECONSTRUCT_NO);
		cudaGaugeField* padded_gauge_field = createExtendedGauge(*gaugePrecise, gR, profile, true, QUDA_RECONSTRUCT_NO);

		DiracParam dirac_param;
		setDiracParam(dirac_param, inv_param, true); // pc = true
		dirac_param.gauge = padded_gauge_field;

		DiracParam dirac_param_precondition;
		setDiracParam(dirac_param_precondition, inv_param, true); // pc = true
		dirac_param_precondition.gauge = padded_gauge_field_precondition;

		for(int i = 0; i < 4; i++){
			dirac_param.commDim[i] = 0;
			dirac_param_precondition.commDim[i] = 0;
		}

		mat = Dirac::create(dirac_param);
		mat_precondition = Dirac::create(dirac_param_precondition);
		dirac_param.print();
		dirac_param_precondition.print();

		MdagM = new DiracMdagM(mat);
//		MdagM = mat;
		MdagM_prec = new DiracMdagM(mat_precondition);
		
		fillInnerSolverParam(solver_prec_param, param);
    
		solver_prec = new CG(*MdagM_prec, *MdagM_prec, solver_prec_param, profile);
	
		printfQuda("MSPCG constructor ends.\n");
	
	}

  MSPCG::~MSPCG(){
    profile.TPSTART(QUDA_PROFILE_FREE);

    if(solver_prec) delete solver_prec;

		delete MdagM;
		delete MdagM_prec;
		delete mat;

    profile.TPSTOP(QUDA_PROFILE_FREE);
  }

	void MSPCG::inner_cg(ColorSpinorField& ix, ColorSpinorField& ib)
	{
		commGlobalReductionSet(false);
		
		blas::zero(ix);
		static cudaColorSpinorField immp(ib);
		static cudaColorSpinorField ir(ib);
		static cudaColorSpinorField ip(ib);
		static cudaColorSpinorField itmp(ib);
	
		double rk2 = blas::norm2(ir);
		double Mpk2, MdagMpk2, alpha, beta, rkp12;

		printfQuda("inner_cg: before starting: r2 = %8.4e \n", rk2);

		for(int local_loop_count = 0; local_loop_count < 10; local_loop_count++){
			(*MdagM_prec)(immp, ip, itmp);
			//zero_boundary_fermion(eg, mmp);
      Mpk2 = reDotProduct(ip, immp);
      MdagMpk2 = blas::norm2(immp); // Dag yes, (Mdag * M * p_k, Mdag * M * p_k)

      alpha = rk2 / Mpk2; 

			axpy(alpha, ip, ix);
			axpy(-alpha, immp, ir);
		
			rkp12 = blas::norm2(ir);

			beta = rkp12 / rk2;
			rk2 = rkp12;
	
			xpay(ir, beta, ip);
        
      printfQuda("inner_cg: l.i. #%04d: r2 = %8.4e alpha = %8.4e beta = %8.4e Mpk2 = %8.4e\n",
      						local_loop_count, rk2, alpha, beta, Mpk2);
		}

		commGlobalReductionSet(true);
		return;
	}

  void MSPCG::operator()(ColorSpinorField& x, ColorSpinorField& b)
  {

    profile.TPSTART(QUDA_PROFILE_INIT);
    // Check to see that we're not trying to invert on a zero-field source
    double b2 = norm2(b);
    if(b2 == 0.){
      profile.TPSTOP(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      x = b;
      param.true_res = 0.;
      param.true_res_hq = 0.;
    }

    int k = 0;

		blas::zero(x);

		int parity = MdagM->getMatPCType();

		ColorSpinorParam csParam(b);
		csParam.create = QUDA_ZERO_FIELD_CREATE;
		// TODO: def
    cudaColorSpinorField* rr = NULL;
    cudaColorSpinorField* tmp = NULL;
    cudaColorSpinorField* mmp = NULL;
    cudaColorSpinorField* z = NULL;
    cudaColorSpinorField* p = NULL;
		
		rr  = new cudaColorSpinorField(csParam);
    tmp = new cudaColorSpinorField(csParam);
    mmp = new cudaColorSpinorField(csParam);
    z   = new cudaColorSpinorField(csParam);
    p   = new cudaColorSpinorField(csParam);

		/// --- 
		printfQuda("Test b2 before = %16.12e.\n", b2);
		printfQuda("rank           = %d.\n", comm_rank());
		if( comm_rank() ){
			blas::zero(b);
		}
    b2 = norm2(b);
		printfQuda("Test b2 after  = %16.12e.\n", b2);
		/// --- 

		cudaColorSpinorField* fr = NULL;
		cudaColorSpinorField* fz = NULL;
		
		cudaColorSpinorField* fx = NULL;
		cudaColorSpinorField* fb = NULL;

		for(int i=0; i<4; ++i){
			csParam.x[i] += 2*R[i];
		}
		printfQuda("[%d %d %d %d %d]\n", csParam.x[0], csParam.x[1], csParam.x[2], csParam.x[3], csParam.x[4]);
	
		csParam.print();

		csParam.setPrecision(QUDA_DOUBLE_PRECISION);
		// TODO: def
		fx  = new cudaColorSpinorField(csParam);
		fb  = new cudaColorSpinorField(csParam);
		
		copyExtendedColorSpinor(*fb, b, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
		((DiracMobius*)mat)->Dslash4pre(*fx, *fb, QUDA_EVEN_PARITY);
		double fx2 = norm2(*fx);
		double fb2 = norm2(*fb);
		printfQuda("Test fx**2/fb**2 = %16.12e/%16.12e.\n", fx2, fb2);
		blas::zero(*fx);

		csParam.setPrecision(QUDA_HALF_PRECISION);
		// TODO: def
		fr  = new cudaColorSpinorField(csParam);
		fz  = new cudaColorSpinorField(csParam);
		
		copyExtendedColorSpinor(*fr, b, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
		((DiracMobius*)mat_precondition)->Dslash4pre(*fz, *fr, QUDA_EVEN_PARITY);
		double sfz2 = norm2(*fz);
		double sfr2 = norm2(*fr);
		printfQuda("Test sfz**2/sfr**2 = %16.12e/%16.12e.\n", sfz2, sfr2);

		(*MdagM)(*rr, x, *tmp); // r = MdagM * x
    double rr2 = xmyNorm(b, *rr); // r = b - MdagM * x
	
    profile.TPSTOP(QUDA_PROFILE_INIT);

		printfQuda("initial r2/b2: %8.4e/%8.4e.\n", rr2, b2);
		// z = M^-1 r
		copyExtendedColorSpinor(*fr, *rr, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
//    (*solver_prec)(*fz, *fr);
	  blas::copy(*fz, *fr);
		copyExtendedColorSpinor(*z, *fz, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);

		blas::copy(*z, *rr);

		double z2 = blas::norm2(*z);
		printfQuda("initial z2 = %8.4e.\n", z2);

    profile.TPSTART(QUDA_PROFILE_COMPUTE);
		// TODO: def
		blas::copy(*p, *z);
		
		double alpha, beta, rkzk, pkApk, zkP1rkp1, zkrk;

		double stop = stopping(param.tol, b2, param.residual_type);
    // TODO: need to fix this heavy quear residual bla bla ...
//		while( r2>stop && k < param.maxiter ){
//		while( k < param.maxiter ){
		while( k < 1 ){
	
			rkzk = reDotProduct(*rr, *z);
			(*MdagM)(*mmp, *p, *tmp);
			pkApk = reDotProduct(*p, *mmp);
			alpha = rkzk / pkApk;

			axpy(alpha, *p, x); // x_k+1 = x_k + alpha * p_k
			axpy(-alpha, *mmp, *rr); // r_k+1 = r_k - alpha * Ap_k
			
    	profile.TPSTOP(QUDA_PROFILE_COMPUTE);
			// z = M^-1 r
			blas::zero(*fr);
			copyExtendedColorSpinor(*fr, *rr, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
//			blas::copy(*fr, *rr);
			double fr2 = blas::norm2(*fr);
//	  (*solver_prec)(*fz, *fr);
	    blas::copy(*fz, *fr);
			blas::zero(*z);
			copyExtendedColorSpinor(*z, *fz, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
			double fz2 = blas::norm2(*fz);
//			blas::copy(*z, *rr);
			z2 = blas::norm2(*z);
			profile.TPSTART(QUDA_PROFILE_COMPUTE);
			
			zkP1rkp1 = reDotProduct(*z, *rr);
			beta = zkP1rkp1 / rkzk;
			xpay(*z, beta, *p);
			
			rr2 = blas::norm2(*rr);
			printfQuda("z2/fz2/r2/fr2: %8.4e/%8.4e/%8.4e/%8.4e\n", z2, fz2, rr2, fr2);
      
			++k;
      // PrintStats("MSPCG", k, r2, b2, 0.);
			printfQuda("MSPCG/iter.count/r2/target_r2/%%/target_%%: %05d %8.4e %8.4e %8.4e %8.4e\n", k, rr2, stop, std::sqrt(rr2/b2), param.tol);
		}
    
		profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + MdagM->flops() + MdagM_prec->flops())*1e-9;
    param.gflops = gflops;
    param.iter += k;

    if (k==param.maxiter)
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    // compute the true residual 
    (*MdagM)(*rr, x, *tmp);
    double true_res = xmyNorm(b, *rr);
    param.true_res = sqrt(true_res/b2);

    // reset the flops counters
    blas::flops = 0;
    (*MdagM).flops();
    (*MdagM_prec).flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTART(QUDA_PROFILE_FREE);

		delete rr  ;
    delete tmp ;
    delete mmp ;
    delete z   ;
    delete p   ;
	
    profile.TPSTOP(QUDA_PROFILE_FREE);
    return;
  }


} // namespace quda
