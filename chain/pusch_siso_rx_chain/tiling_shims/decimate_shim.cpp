
#define GenerateTiling   gt_decimate
#define GetTilingSize    gts_decimate
#define GetWorkspaceSize gws_decimate
#include "../../../sync/decimate/decimate_tiling.cpp"
#undef GenerateTiling
#undef GetTilingSize
#undef GetWorkspaceSize
