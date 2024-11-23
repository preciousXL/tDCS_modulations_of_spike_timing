#include <stdio.h>
#include "hocdec.h"
#define IMPORT extern __declspec(dllimport)
IMPORT int nrnmpi_myid, nrn_nobanner_;

extern void _SlowCa_reg();
extern void _cad2_reg();
extern void _child_reg();
extern void _childa_reg();
extern void _epsp_reg();
extern void _it2_reg();
extern void _kaprox_reg();
extern void _kca_reg();
extern void _km_reg();
extern void _kv_reg();
extern void _na_reg();
extern void _xtra_reg();

void modl_reg(){
	//nrn_mswindll_stdio(stdin, stdout, stderr);
    if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
	fprintf(stderr, "Additional mechanisms from files\n");

fprintf(stderr," SlowCa.mod");
fprintf(stderr," cad2.mod");
fprintf(stderr," child.mod");
fprintf(stderr," childa.mod");
fprintf(stderr," epsp.mod");
fprintf(stderr," it2.mod");
fprintf(stderr," kaprox.mod");
fprintf(stderr," kca.mod");
fprintf(stderr," km.mod");
fprintf(stderr," kv.mod");
fprintf(stderr," na.mod");
fprintf(stderr," xtra.mod");
fprintf(stderr, "\n");
    }
_SlowCa_reg();
_cad2_reg();
_child_reg();
_childa_reg();
_epsp_reg();
_it2_reg();
_kaprox_reg();
_kca_reg();
_km_reg();
_kv_reg();
_na_reg();
_xtra_reg();
}
