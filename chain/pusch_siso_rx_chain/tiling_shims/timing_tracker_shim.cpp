
#define GenerateTiling   gt_timing_tracker
#define GetTilingSize    gts_timing_tracker
#define GetWorkspaceSize gws_timing_tracker
#include "../../../sync/timing_tracker/timing_tracker_tiling.cpp"
#undef GenerateTiling
#undef GetTilingSize
#undef GetWorkspaceSize
