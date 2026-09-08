
#define GenerateTiling   gt_cfo_compensate
#define GetTilingSize    gts_cfo_compensate
#define GetWorkspaceSize gws_cfo_compensate
#include "../../../sync/cfo_compensate/cfo_compensate_tiling.cpp"
#undef GenerateTiling
#undef GetTilingSize
#undef GetWorkspaceSize
