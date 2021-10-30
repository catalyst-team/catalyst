from functools import partial

from catalyst import dl, SETTINGS

E2E = {
    "de": dl.DeviceEngine,
    "dp": dl.DataParallelEngine,
    "ddp": dl.DistributedDataParallelEngine,
}

if SETTINGS.amp_required:
    E2E.update(
        {"amp-dp": dl.DataParallelAMPEngine, "amp-ddp": dl.DistributedDataParallelAMPEngine}
    )

if SETTINGS.apex_required:
    E2E.update(
        {"apex-dp": dl.DataParallelAPEXEngine, "apex-ddp": dl.DistributedDataParallelAPEXEngine}
    )

if SETTINGS.deepspeed_required:
    E2E.update({"ds-ddp": dl.DistributedDataParallelDeepSpeedEngine})

if SETTINGS.fairscale_required:
    E2E.update(
        {
            "fs-pp": dl.PipelineParallelFairScaleEngine,
            "fs-ddp": dl.SharedDataParallelFairScaleEngine,
            "fs-ddp-amp": dl.SharedDataParallelFairScaleAMPEngine,
            # for some reason we could catch a bug with FairScale flatten wrapper here, so...
            "fs-fddp": partial(
                dl.FullySharedDataParallelFairScaleEngine, ddp_kwargs={"flatten_parameters": False}
            ),
        }
    )

if SETTINGS.xla_required:
    E2E.update({"xla": dl.XLAEngine, "xla-ddp": dl.DistributedXLAEngine})
