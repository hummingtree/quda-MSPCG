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
    inner.tol = outer.tol_precondition;
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
    Solver(_param, profile), solver_prec(0), solver_prec_param(_param)
  { // note that for consistency will accept ONLY M and WILL promote that to MdagM.
   
		R.fill(ps);

		if(inv_param->dslash_type != QUDA_MOBIUS_DWF_DSLASH){
			errorQuda("ONLY works for QUDA_MOBIUS_DWF_DSLASH.");
		}

		// create extended gauge field
		// TODO: dynamical allocation need fix
		cudaGaugeField* padded_gauge_field = createExtendedGauge(*gaugePrecondition, R.data(), profile, true, QUDA_RECONSTRUCT_INVALID);

		DiracParam dirac_param;
		setDiracParam(dirac_param, inv_param, true); // pc = true

		DiracParam dirac_prec_param;
		setDiracParam(dirac_prec_param, inv_param, true); // pc = true
		dirac_prec_param.gauge = padded_gauge_field;

		for(int i = 0; i < 4; i++){
			dirac_prec_param.commDim[i] = 0;
		}

		Dirac* mat = Dirac::create(dirac_param);
		Dirac* mat_prec = Dirac::create(dirac_prec_param);

		MdagM = new DiracMdagM(mat);
		MdagM_prec = new DiracMdagM(mat_prec);

		
		fillInnerSolverParam(solver_prec_param, param);
    
		solver_prec = new CG(*MdagM_prec, *MdagM_prec, solver_prec_param, profile);
	
	}

  MSPCG::~MSPCG(){
    profile.TPSTART(QUDA_PROFILE_FREE);

    if(solver_prec) delete solver_prec;

		delete MdagM;
		delete MdagM_prec;

    profile.TPSTOP(QUDA_PROFILE_FREE);
  }


  void MSPCG::operator()(ColorSpinorField& x, ColorSpinorField& b)
  {

    profile.TPSTART(QUDA_PROFILE_INIT);
    // Check to see that we're not trying to invert on a zero-field source
    const double b2 = norm2(b);
    if(b2 == 0.){
      profile.TPSTOP(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      x = b;
      param.true_res = 0.;
      param.true_res_hq = 0.;
    }

    int k = 0;

		int parity = MdagM->getMatPCType();

		ColorSpinorParam csParam(b);
		csParam.create = QUDA_ZERO_FIELD_CREATE;
		// TODO: def
    cudaColorSpinorField r(b, csParam);
    cudaColorSpinorField tmp(b, csParam);
    cudaColorSpinorField mmp(b, csParam);
    cudaColorSpinorField z(b, csParam);
		
		cudaColorSpinorField* fr = NULL;
		cudaColorSpinorField* fz = NULL;

		for(int i=0; i<4; ++i) csParam.x[i] += 2*R[i];
		// TODO: def
		fr = new cudaColorSpinorField(csParam);
		fz = new cudaColorSpinorField(csParam);
		
		(*MdagM)(r, x, tmp); // r = MdagM * x
    double r2 = xmyNorm(b, r); // r = b - MdagM * x

		// z = M^-1 r
		copyExtendedColorSpinor(*fr, r, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
    (*solver_prec)(*fz, *fr);
		copyExtendedColorSpinor(z, *fz, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);

		// TODO: def
		cudaColorSpinorField p(z);
	
		double alpha, beta, rkzk, pkApk, zkP1rkp1, zkrk;

		double stop = stopping(param.tol, b2, param.residual_type);
    // TODO: need to fix this heavy quear residual bla bla ...
		while(!convergence(r2, 0., stop, param.tol_hq) && k < param.maxiter){
			
			rkzk = reDotProduct(r, z);
			(*MdagM)(mmp, p, tmp);
			pkApk = reDotProduct(p, mmp);
			alpha = rkzk / pkApk;

			axpy(alpha, p, x); // x_k+1 = x_k + alpha * p_k
			axpy(-alpha, mmp, r); // r_k+1 = r_k - alpha * Ap_k
			
			// z = M^-1 r
			copyExtendedColorSpinor(*fr, r, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);
	    (*solver_prec)(*fz, *fr);
			copyExtendedColorSpinor(z, *fz, QUDA_CUDA_FIELD_LOCATION, parity, NULL, NULL, NULL, NULL);

			zkP1rkp1 = reDotProduct(z, r);
			beta = zkP1rkp1 / rkzk;
			xpay(z, beta, p);
			
			r2 = blas::norm2(r);
      
			++k;
      PrintStats("MSPCG", k, r2, b2, 0.);

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
    (*MdagM)(r, x, tmp);
    double true_res = xmyNorm(b, r);
    param.true_res = sqrt(true_res/b2);

    // reset the flops counters
    blas::flops = 0;
    (*MdagM).flops();
    (*MdagM_prec).flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTART(QUDA_PROFILE_FREE);

    profile.TPSTOP(QUDA_PROFILE_FREE);
    return;
  }


} // namespace quda
