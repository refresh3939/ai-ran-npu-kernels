#define GenerateTiling   gt_scramble
#define GetTilingSize    gts_scramble
#define GetWorkspaceSize gws_scramble
#include "../../../fec/scramble_siso/scramble_tiling.cpp"
#undef GenerateTiling
#undef GetTilingSize
#undef GetWorkspaceSize
