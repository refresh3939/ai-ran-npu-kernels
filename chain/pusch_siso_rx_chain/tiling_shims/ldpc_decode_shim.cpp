
#define GenerateTiling   gt_ldpc_decode
#define GetTilingSize    gts_ldpc_decode
#define GetWorkspaceSize gws_ldpc_decode
#include "../../../fec/ldpc_decode/ldpc_decode_tiling.cpp"
#undef GenerateTiling
#undef GetTilingSize
#undef GetWorkspaceSize
