#define GenerateTiling   gt_qam256_mod
#define GetTilingSize    gts_qam256_mod
#define GetWorkspaceSize gws_qam256_mod
#include "../../../mapping/qam_mod_256_siso/qam256_mod_tiling.cpp"
#undef GenerateTiling
#undef GetTilingSize
#undef GetWorkspaceSize
