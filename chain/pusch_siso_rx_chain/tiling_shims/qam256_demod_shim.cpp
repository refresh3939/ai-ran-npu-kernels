
#define GenerateTiling   gt_qam256_demod
#define GetTilingSize    gts_qam256_demod
#define GetWorkspaceSize gws_qam256_demod
#include "../../../mapping/qam_demod_256_siso/qam256_demod_tiling.cpp"
#undef GenerateTiling
#undef GetTilingSize
#undef GetWorkspaceSize
