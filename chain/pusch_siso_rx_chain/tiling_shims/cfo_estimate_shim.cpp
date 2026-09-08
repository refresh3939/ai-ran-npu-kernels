
#define GenerateTiling   gt_cfo_estimate
#define GetTilingSize    gts_cfo_estimate
#define GetWorkspaceSize gws_cfo_estimate
#include "../../../sync/cfo_estimate/cfo_estimate_tiling.cpp"
#undef GenerateTiling
#undef GetTilingSize
#undef GetWorkspaceSize
