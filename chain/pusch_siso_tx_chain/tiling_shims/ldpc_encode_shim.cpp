#define GenerateTiling   gt_ldpc_encode
#define GetTilingSize    gts_ldpc_encode
#define GetWorkspaceSize gws_ldpc_encode
#include "../../../fec/ldpc_encode/ldpc_encode_tiling.cpp"
#undef GenerateTiling
#undef GetTilingSize
#undef GetWorkspaceSize
