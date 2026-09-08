
#define GenerateTiling   gt_equalize
#define GetTilingSize    gts_equalize
#define GetWorkspaceSize gws_equalize
#include "../../../equalization/equalize_zf_siso/equalize_tiling.cpp"
#undef GenerateTiling
#undef GetTilingSize
#undef GetWorkspaceSize
