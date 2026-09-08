#define GenerateTiling   gt_rate_match
#define GetTilingSize    gts_rate_match
#define GetWorkspaceSize gws_rate_match
#include "../../../fec/rate_match_siso/rate_match_tiling.cpp"
#undef GenerateTiling
#undef GetTilingSize
#undef GetWorkspaceSize
