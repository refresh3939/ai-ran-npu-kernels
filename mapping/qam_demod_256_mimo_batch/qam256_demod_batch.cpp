#include "qam256_demod_batch.h"

#include "acl/acl.h"
#include "aclrtlaunch_qam256_demod_batch_kernel.h"

namespace airan::qam256_demod_batch {
Status Enqueue(const QamDemod256BatchOpArgsV1 &args,
               void *workspace,
               size_t workspace_bytes,
               const void *tiling,
               size_t tiling_bytes)
{
    const Status status = ValidateOpArgs(args);
    if (status != OK) return status;
    if (workspace == nullptr || tiling == nullptr ||
        workspace_bytes < WORKSPACE_BYTES || tiling_bytes < TILING_BYTES) {
        return RESOURCE_TOO_SMALL;
    }

    const auto stream = static_cast<aclrtStream>(args.stream);

    const uint32_t launch_status = ACLRT_LAUNCH_KERNEL(qam256_demod_batch_kernel)(
        BLOCK_DIM, stream,
        const_cast<void *>(args.x_re), const_cast<void *>(args.x_im),
        const_cast<void *>(args.no_eff), args.layer_llr,
        workspace, const_cast<void *>(tiling));
    return launch_status == ACL_ERROR_NONE ? OK : LAUNCH_FAILED;
}

}
