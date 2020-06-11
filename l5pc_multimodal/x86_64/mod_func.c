#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _CaDynamics_E2_reg(void);
extern void _Ca_HVA_reg(void);
extern void _Ca_LVAst_reg(void);
extern void _Ih_reg(void);
extern void _Im_reg(void);
extern void _K_Pst_reg(void);
extern void _K_Tst_reg(void);
extern void _NaTa_t_reg(void);
extern void _NaTs2_t_reg(void);
extern void _Nap_Et2_reg(void);
extern void _ProbAMPANMDA_EMS_reg(void);
extern void _ProbGABAAB_EMS_reg(void);
extern void _SK_E2_reg(void);
extern void _SKv3_1_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," mechanisms/CaDynamics_E2.mod");
    fprintf(stderr," mechanisms/Ca_HVA.mod");
    fprintf(stderr," mechanisms/Ca_LVAst.mod");
    fprintf(stderr," mechanisms/Ih.mod");
    fprintf(stderr," mechanisms/Im.mod");
    fprintf(stderr," mechanisms/K_Pst.mod");
    fprintf(stderr," mechanisms/K_Tst.mod");
    fprintf(stderr," mechanisms/NaTa_t.mod");
    fprintf(stderr," mechanisms/NaTs2_t.mod");
    fprintf(stderr," mechanisms/Nap_Et2.mod");
    fprintf(stderr," mechanisms/ProbAMPANMDA_EMS.mod");
    fprintf(stderr," mechanisms/ProbGABAAB_EMS.mod");
    fprintf(stderr," mechanisms/SK_E2.mod");
    fprintf(stderr," mechanisms/SKv3_1.mod");
    fprintf(stderr, "\n");
  }
  _CaDynamics_E2_reg();
  _Ca_HVA_reg();
  _Ca_LVAst_reg();
  _Ih_reg();
  _Im_reg();
  _K_Pst_reg();
  _K_Tst_reg();
  _NaTa_t_reg();
  _NaTs2_t_reg();
  _Nap_Et2_reg();
  _ProbAMPANMDA_EMS_reg();
  _ProbGABAAB_EMS_reg();
  _SK_E2_reg();
  _SKv3_1_reg();
}
