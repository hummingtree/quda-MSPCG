#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <test_util.h>
#include <dslash_util.h>
#include <blas_reference.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>
#include "misc.h"

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <qio_field.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

#include <mspcg.h>

#include <qlat/qlat.h>

// Wilson, clover-improved Wilson, twisted mass, and domain wall are supported.
extern QudaDslashType dslash_type;

// Twisted mass flavor type
extern QudaTwistFlavorType twist_flavor;

extern int device;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int Lsdim;
extern int gridsize_from_cmdline[];
extern QudaPrecision prec;
extern QudaPrecision  prec_sloppy;
extern QudaPrecision  prec_precondition;
extern QudaReconstructType link_recon;
extern QudaReconstructType link_recon_sloppy;
extern QudaReconstructType link_recon_precondition;
extern QudaInverterType  inv_type;
extern QudaInverterType  precon_type;
extern int multishift; // whether to test multi-shift or standard solver
extern double mass; // mass of Dirac operator
extern double mu;
extern double anisotropy; // temporal anisotropy
extern double tol; // tolerance for inverter
extern double tol_hq; // heavy-quark tolerance for inverter
extern QudaMassNormalization normalization; // mass normalization of Dirac operators
extern QudaMatPCType matpc_type; // preconditioning type

extern double clover_coeff;
extern bool compute_clover;

extern int niter; // max solver iterations
extern int gcrNkrylov; // number of inner iterations for GCR, or l for BiCGstab-l
extern int pipeline; // length of pipeline for fused operations in GCR or BiCGstab-l
extern int solution_accumulator_pipeline; // length of pipeline for fused solution update from the direction vectors
extern char latfile[];

extern void usage(char** );

using namespace quda;

quda::cudaGaugeField* checkGauge(QudaInvertParam *param);

extern quda::cudaGaugeField* gaugePrecise;

//!< Profiler for invertQuda
static TimeProfile profileInvert("invertQuda");

static bool initialized = false;
static bool comms_initialized = false;

  void
display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    prec_sloppy   multishift  matpc_type  recon  recon_sloppy S_dimension T_dimension Ls_dimension   dslash_type  normalization\n");
  printfQuda("%6s   %6s          %d     %12s     %2s     %2s         %3d/%3d/%3d     %3d         %2d       %14s  %8s\n",
      get_prec_str(prec),get_prec_str(prec_sloppy), multishift, get_matpc_str(matpc_type), 
      get_recon_str(link_recon), 
      get_recon_str(link_recon_sloppy),  
      xdim, ydim, zdim, tdim, Lsdim, 
      get_dslash_str(dslash_type), 
      get_mass_normalization_str(normalization));     

  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
      dimPartitioned(0),
      dimPartitioned(1),
      dimPartitioned(2),
      dimPartitioned(3)); 

  return ;

}

void invert_MSPCG(void *hp_x, void *hp_b, QudaInvertParam *param)
{
  setKernelPackT(true);

  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);

  //  if (!initialized) errorQuda("QUDA not initialized");

  printQudaInvertParam(param);

  //  checkInvertParam(param);

  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(param);
  //  cudaGaugeField *cudaGauge = gaugePrecise;

  // It was probably a bad design decision to encode whether the system is even/odd preconditioned (PC) in
  // solve_type and solution_type, rather than in separate members of QudaInvertParam.  We're stuck with it
  // for now, though, so here we factorize everything for convenience.

  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) ||
    (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
    (param->solve_type == QUDA_NORMOP_PC_SOLVE) || (param->solve_type == QUDA_NORMERR_PC_SOLVE);
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) ||
    (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (param->solve_type == QUDA_DIRECT_SOLVE) ||
    (param->solve_type == QUDA_DIRECT_PC_SOLVE);
  bool norm_error_solve = (param->solve_type == QUDA_NORMERR_SOLVE) ||
    (param->solve_type == QUDA_NORMERR_PC_SOLVE);

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }

  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;

  profileInvert.TPSTART(QUDA_PROFILE_H2D);

  ColorSpinorField *b = NULL;
  ColorSpinorField *x = NULL;
  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;

  const int *X = cudaGauge->X();

  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(hp_b, *param, X, pc_solution, param->input_location);
  ColorSpinorField *h_b = ColorSpinorField::Create(cpuParam);

  cpuParam.v = hp_x;
  cpuParam.location = param->output_location;
  ColorSpinorField *h_x = ColorSpinorField::Create(cpuParam);

  // download source
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_COPY_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam);

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);

  blas::zero(*x);

  profileInvert.TPSTOP(QUDA_PROFILE_H2D);

  double nb = blas::norm2(*b);
  if (nb==0.) errorQuda("Source has zero norm");

  double nh_b = blas::norm2(*h_b);
  double nh_x = blas::norm2(*h_x);
  double nx = blas::norm2(*x);
  printfQuda("Source: CPU = %g, CUDA copy = %g\n", nh_b, nb);
  printfQuda("Solution: CPU = %g, CUDA copy = %g\n", nh_x, nx);

  in = b;
  out = x;

  double nin = blas::norm2(*in);
  double nout = blas::norm2(*out);
  printfQuda("Prepared source = %g\n", nin);
  printfQuda("Prepared solution = %g\n", nout);

  // Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
  // (*solve)(*out, *in);
  // solverParam.updateInvertParam(*param);
  // delete solve;

  SolverParam solverParam(*param);

  MSPCG* mspcg = new MSPCG(param, solverParam, profileInvert);
  (*mspcg)(*out, *in);
  solverParam.updateInvertParam(*param);
  delete mspcg;

  nx = blas::norm2(*x);
  printfQuda("Solution = %g\n",nx);

  profileInvert.TPSTART(QUDA_PROFILE_EPILOGUE);

  profileInvert.TPSTOP(QUDA_PROFILE_EPILOGUE);

  profileInvert.TPSTART(QUDA_PROFILE_D2H);
  *h_x = *x;
  profileInvert.TPSTOP(QUDA_PROFILE_D2H);

  profileInvert.TPSTART(QUDA_PROFILE_EPILOGUE);

  nx = blas::norm2(*x);
  nh_x = blas::norm2(*h_x);
  printfQuda("Reconstructed: CUDA solution = %g, CPU copy = %g\n", nx, nh_x);

  profileInvert.TPSTOP(QUDA_PROFILE_EPILOGUE);

  profileInvert.TPSTART(QUDA_PROFILE_FREE);

  delete x;
  delete h_b;
  delete h_x;
  delete b;

  profileInvert.TPSTOP(QUDA_PROFILE_FREE);

  //  popVerbosity();

  // cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache();

  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);
}

int main(int argc, char **argv)
{

  for (int i = 1; i < argc; i++){
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    } 
    printfQuda("ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

// qlat initialization
	qlat::Coordinate node_coor( commCoords(0), commCoords(1), commCoords(2), commCoords(3) );
	qlat::Coordinate node_size( commDim(0), commDim(1), commDim(2), commDim(3) );
	qlat::begin(qlat::index_from_coordinate(node_coor, node_size), node_size);
	printf("Node #%03d(quda):%02dx%02dx%02dx%02d/#%03d(qlat):%02dx%02dx%02dx%02d.\n", comm_rank(), 
				commCoords(0), commCoords(1), commCoords(2), commCoords(3), 
				qlat::get_id_node(), qlat::get_coor_node()[0], qlat::get_coor_node()[1], qlat::get_coor_node()[2], qlat::get_coor_node()[3]);
// END qlat initialization

  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (prec_precondition == QUDA_INVALID_PRECISION) prec_precondition = prec_sloppy;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;
  if (link_recon_precondition == QUDA_RECONSTRUCT_INVALID) link_recon_precondition = link_recon_sloppy;

  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  display_test_info();

  // *** QUDA parameters begin here.

  QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
  QudaPrecision cuda_prec = prec;
  QudaPrecision cuda_prec_sloppy = prec_sloppy;
  QudaPrecision cuda_prec_precondition = prec_precondition;

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();

  double kappa5;

  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;
  inv_param.Ls = 1;

  gauge_param.anisotropy = anisotropy;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_PERIODIC_T;

  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = link_recon;
  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;
  gauge_param.cuda_prec_precondition = cuda_prec_precondition;
  gauge_param.reconstruct_precondition = link_recon_precondition;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  inv_param.dslash_type = dslash_type;

  inv_param.mass = mass;
  inv_param.mu = mu;
  inv_param.kappa = 1.0 / (2.0 * (1 + 3/gauge_param.anisotropy + mass));

  if ( dslash_type != QUDA_MOBIUS_DWF_DSLASH ) printfQuda("ERROR: NOR Mobius?\n");

  inv_param.m5 = -1.8;
  kappa5 = 0.5/(5 + inv_param.m5);  
  inv_param.Ls = Lsdim;
  for(int k = 0; k < Lsdim; k++){
    // b5[k], c[k] values are chosen for arbitrary values,
    // but the difference of them are same as 1.0
    inv_param.b_5[k] = 1.452;
    inv_param.c_5[k] = 0.452;
  }


  inv_param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;

  inv_param.matpc_type = matpc_type;

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = normalization;
  inv_param.solver_normalization = QUDA_DEFAULT_NORMALIZATION;

  inv_param.solve_type = QUDA_NORMOP_PC_SOLVE;

  inv_param.pipeline = pipeline;

  inv_param.Nsteps = 2;
  inv_param.gcrNkrylov = gcrNkrylov;
  inv_param.tol = tol;
  inv_param.tol_restart = 1e-3; //now theoretical background for this parameter... 
  if(tol_hq == 0 && tol == 0){
    errorQuda("qudaInvert: requesting zero residual\n");
    exit(1);
  }

  // require both L2 relative and heavy quark residual to determine convergence
  inv_param.residual_type = static_cast<QudaResidualType_s>(0);
  inv_param.residual_type = (tol != 0) ? static_cast<QudaResidualType_s> ( inv_param.residual_type | QUDA_L2_RELATIVE_RESIDUAL) : inv_param.residual_type;
  inv_param.residual_type = (tol_hq != 0) ? static_cast<QudaResidualType_s> (inv_param.residual_type | QUDA_HEAVY_QUARK_RESIDUAL) : inv_param.residual_type;

  inv_param.tol_hq = tol_hq; // specify a tolerance for the residual for heavy quark residual

  // these can be set individually
  for (int i=0; i<inv_param.num_offset; i++) {
    inv_param.tol_offset[i] = inv_param.tol;
    inv_param.tol_hq_offset[i] = inv_param.tol_hq;
  }
  inv_param.maxiter = niter;
  inv_param.reliable_delta = 1e-1;
  inv_param.use_sloppy_partial_accumulator = 0;
  inv_param.solution_accumulator_pipeline = solution_accumulator_pipeline;
  inv_param.max_res_increase = 1;

  // domain decomposition preconditioner parameters
  inv_param.inv_type_precondition = precon_type;

  inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
  inv_param.precondition_cycle = 1;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 10;
  inv_param.verbosity_precondition = QUDA_DEBUG_VERBOSE;
  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.omega = 1.0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_YES;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  gauge_param.ga_pad = 0; // 24*24*24/2;
  inv_param.sp_pad = 0; // 24*24*24/2;
  inv_param.cl_pad = 0; // 24*24*24/2;

  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;    
#endif

  inv_param.verbosity = QUDA_DEBUG_VERBOSE;

  // *** Everything between here and the call to initQuda() is
  // *** application-specific.

  // set parameters for the reference Dslash, and prepare fields to be loaded
  dw_setDims(gauge_param.X, inv_param.Ls);

  setSpinorSiteSize(24);

  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  void *gauge[4], *clover=0, *clover_inv=0;

  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*gSize);
  }

  if (strcmp(latfile,"")) {  // load in the command line supplied gauge field
    read_gauge_field(latfile, gauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    construct_gauge_field(gauge, 2, gauge_param.cpu_prec, &gauge_param);
  } else { // else generate a random SU(3) field
    construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);
  }

  void *spinorIn = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
  void *spinorCheck = malloc(V*spinorSiteSize*sSize*inv_param.Ls);

  void *spinorOut = NULL, **spinorOutMulti = NULL;
  spinorOut = malloc(V*spinorSiteSize*sSize*inv_param.Ls);

  memset(spinorIn, 0, inv_param.Ls*V*spinorSiteSize*sSize);
  memset(spinorCheck, 0, inv_param.Ls*V*spinorSiteSize*sSize);

  memset(spinorOut, 0, inv_param.Ls*V*spinorSiteSize*sSize);

  // create a point source at 0 (in each subvolume...  FIXME)

  if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION) {
    for (int i=0; i<inv_param.Ls*V*spinorSiteSize; i++) ((float*)spinorIn)[i] = rand() / (float)RAND_MAX;
  }else{
    //    for (int i=0; i<inv_param.Ls*V*spinorSiteSize; i++) ((double*)spinorIn)[i] = rand() / (double)RAND_MAX;
    for (int i=0; i<inv_param.Ls*V*spinorSiteSize; i++) ((double*)spinorIn)[i] = double(i%23);
    //    for (int i=0; i<inv_param.Ls*V*spinorSiteSize; i++) ((double*)spinorIn)[i] = 1.;
  }

  // start the timer
  double time0 = -((double)clock());

  // initialize the QUDA library
  initQuda(device);

  // load the gauge field
  loadGaugeQuda((void*)gauge, &gauge_param);

  // perform the inversion
  invert_MSPCG(spinorOut, spinorIn, &inv_param);

  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;

  printfQuda("Device memory used:\n   Spinor: %f GiB\n    Gauge: %f GiB\n", 
      inv_param.spinorGiB, gauge_param.gaugeGiB);

  printfQuda("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n", 
      inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, time0);

  void *spinorTmp = malloc(V*spinorSiteSize*sSize*inv_param.Ls);

  double *kappa_b, *kappa_c;
  kappa_b = (double*)malloc(Lsdim*sizeof(double));
  kappa_c = (double*)malloc(Lsdim*sizeof(double));
  for(int xs = 0 ; xs < Lsdim ; xs++)
  {
    kappa_b[xs] = 1.0/(2*(inv_param.b_5[xs]*(4.0 + inv_param.m5) + 1.0));
    kappa_c[xs] = 1.0/(2*(inv_param.c_5[xs]*(4.0 + inv_param.m5) - 1.0));
  }
  mdw_matpc(spinorTmp, gauge, spinorOut, kappa_b, kappa_c, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
  mdw_matpc(spinorCheck, gauge, spinorTmp, kappa_b, kappa_c, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
  free(kappa_b);
  free(kappa_c);

  int vol = inv_param.solution_type == QUDA_MAT_SOLUTION ? V : Vh;
  mxpy(spinorIn, spinorCheck, vol*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
  double nrm2 = norm_2(spinorCheck, vol*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
  double src2 = norm_2(spinorIn, vol*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
  double l2r = sqrt(nrm2 / src2);

  printfQuda("Residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g\n",
      inv_param.tol, inv_param.true_res, l2r, inv_param.tol_hq, inv_param.true_res_hq);

  freeGaugeQuda();

  // finalize the QUDA library
  endQuda();
  finalizeComms();

  for (int dir = 0; dir<4; dir++) free(gauge[dir]);

  return 0;
}
