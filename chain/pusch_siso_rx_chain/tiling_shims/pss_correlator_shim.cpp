
#define GenerateTiling   gt_pss_correlator
#define GetTilingSize    gts_pss_correlator
#define GetWorkspaceSize gws_pss_correlator
#include "../../../sync/pss_correlator/pss_correlator_tiling.cpp"
#undef GenerateTiling
#undef GetTilingSize
#undef GetWorkspaceSize
