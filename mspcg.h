#ifndef _MSPCG_H
#define _MSPCG_H

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <color_spinor_field.h>
#include <vector>

#include <invert_quda.h>

namespace quda {

	class MSPCG : public Solver { // Multisplitting Preconditioned CG
	
	private:
		
		DiracMdagM* MdagM;
		DiracMdagM* MdagM_prec;

		Solver *solver_prec;
		SolverParam solver_prec_param;
		
		std::array<int, 4> R;

	public:
		
		MSPCG(QudaInvertParam* inv_param, SolverParam& _param, TimeProfile& profile, int ps=1);
		
		virtual ~MSPCG();

		void operator()(ColorSpinorField& out, ColorSpinorField& in);

	};

}

#endif
