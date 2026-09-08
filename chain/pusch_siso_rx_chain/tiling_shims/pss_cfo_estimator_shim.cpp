
#define GenerateTiling   gt_pss_cfo_estimator
#define GetTilingSize    gts_pss_cfo_estimator
#define GetWorkspaceSize gws_pss_cfo_estimator
#include "../../../sync/pss_cfo_estimator/pss_cfo_estimator_tiling.cpp"
#undef GenerateTiling
#undef GetTilingSize
#undef GetWorkspaceSize
