#ifndef _MSPCG_H
#define _MSPCG_H

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <color_spinor_field.h>
#include <vector>

namespace quda {

	class MSPCG : public Solver { // Multisplitting Preconditioned CG
	
	private:
		
		const DiracMdagM &MdagM;
		const DiracMdagM &MdagM_prec;

		Solver *solver_prec;
		SolverParam solver_prec_param;
		
		int padding_size;

		ColorSpinorField p;	
		ColorSpinorField r;	
		ColorSpinorField z;	
		ColorSpinorField fz;	
		ColorSpinorField fr;	

	public:
		
		MSPCG(DiracMatrix& _mat, DiracMatrix& _mat_prec, SolverParam& _param, TimeProfile& profile);
		
		virtual ~MSPCG();

		void operator()(ColorSpinorField& out, ColorSpinorField& in);

	};

}

#endif
