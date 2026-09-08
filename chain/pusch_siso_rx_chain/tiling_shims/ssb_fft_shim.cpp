
#define GenerateTiling   gt_ssb_fft
#define GetTilingSize    gts_ssb_fft
#define GetWorkspaceSize gws_ssb_fft
#include "../../../sync/ssb_fft/ssb_fft_tiling.cpp"
#undef GenerateTiling
#undef GetTilingSize
#undef GetWorkspaceSize
