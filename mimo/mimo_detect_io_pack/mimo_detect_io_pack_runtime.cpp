#include "mimo_detect_io_pack.h"

#include "acl/acl.h"
#include "aclrtlaunch_mimo_detect_io_pack_kernel.h"

namespace airan::mimo_detect_io_pack {

Status Enqueue(const MimoDetectIoPackOpArgsV1 &args,
               const void *metadata,
               size_t metadata_bytes)
{
    const Status status = ValidateOpArgs(args);
    if (status != OK) return status;
    if (metadata == nullptr || metadata_bytes < TILING_BYTES) {
        return RESOURCE_TOO_SMALL;
    }

    const auto stream = static_cast<aclrtStream>(args.stream);
    const uint32_t launch_status =
        ACLRT_LAUNCH_KERNEL(mimo_detect_io_pack_kernel)(
            BLOCK_DIM, stream,
            const_cast<void *>(args.rx_grid_re),
            const_cast<void *>(args.rx_grid_im),
            const_cast<void *>(args.h_grid_re),
            const_cast<void *>(args.h_grid_im),
            const_cast<void *>(args.noise_var_rx),
            args.hrm_re, args.hrm_im, args.yvpad_re, args.yvpad_im, args.no,
            nullptr, const_cast<void *>(metadata));
    return launch_status == ACL_ERROR_NONE ? OK : LAUNCH_FAILED;
}

}
