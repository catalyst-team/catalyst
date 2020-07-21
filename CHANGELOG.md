# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [20.07.1] - YYYY-MM-DD

### Added

- `CMCScoreCallback` ([#880](https://github.com/catalyst-team/catalyst/pull/880))
- kornia augmentations `BatchTransformCallback` ([#862](https://github.com/catalyst-team/catalyst/issues/862))
- `average_precision` and `mean_average_precision` metrics ([#883](https://github.com/catalyst-team/catalyst/pull/883))
- `MultiLabelAccuracyCallback`, `AveragePrecisionCallback` and `MeanAveragePrecisionCallback` callbacks ([#883](https://github.com/catalyst-team/catalyst/pull/883))
- minimal examples for multi-class and milti-label classification ([#883](https://github.com/catalyst-team/catalyst/pull/883))

### Changed

- all registries merged to one `catalyst.registry` ([#883](https://github.com/catalyst-team/catalyst/pull/883))

### Removed

- `average_accuracy` and `mean_average_accuracy` metrics ([#883](https://github.com/catalyst-team/catalyst/pull/883))

### Fixed

- `utils.tokenize_text` typo with punctuation ([#880](https://github.com/catalyst-team/catalyst/pull/880))


## [20.07] - 2020-07-06

### Added

- `log` parameter to `WandbLogger` ([#836](https://github.com/catalyst-team/catalyst/pull/836))
- hparams experiment property ([#839](https://github.com/catalyst-team/catalyst/pull/839))
- add docs build on push to master branch ([#844](https://github.com/catalyst-team/catalyst/pull/844))
- `WrapperCallback` and `ControlFlowCallback` ([#842](https://github.com/catalyst-team/catalyst/pull/842))
- `BatchOverfitCallback` ([#869](https://github.com/catalyst-team/catalyst/pull/869))
- `overfit` flag for Config API ([#869](https://github.com/catalyst-team/catalyst/pull/869))
- `InBatchSamplers`: `AllTripletsSampler` and `HardTripletsSampler` ([#825](https://github.com/catalyst-team/catalyst/pull/825))

### Changed

- Renaming ([#837](https://github.com/catalyst-team/catalyst/pull/837))
    - `SqueezeAndExcitation` -> `cSE`
    - `ChannelSqueezeAndSpatialExcitation` -> `sSE`
    - `ConcurrentSpatialAndChannelSqueezeAndChannelExcitation` -> `scSE`
    - `_MetricCallback` -> `IMetricCallback`
    - `dl.Experiment.process_loaders` -> `dl.Experiment._get_loaders`
- `LRUpdater` become abstract class ([#837](https://github.com/catalyst-team/catalyst/pull/837))
- `calculate_confusion_matrix_from_arrays` changed params order ([#837](https://github.com/catalyst-team/catalyst/pull/837))
- `dl.Runner.predict_loader` uses `_prepare_inner_state` and cleans `experiment` ([#863](https://github.com/catalyst-team/catalyst/pull/863))
- `toml` to the dependencies ([#872](https://github.com/catalyst-team/catalyst/pull/872))

### Removed

- `crc32c` dependency ([#872](https://github.com/catalyst-team/catalyst/pull/872))

### Fixed

- `workflows/deploy_push.yml` failed to push some refs ([#864](https://github.com/catalyst-team/catalyst/pull/864))
- `.dependabot/config.yml` contained invalid details ([#781](https://github.com/catalyst-team/catalyst/issues/781))
- `LanguageModelingDataset` ([#841](https://github.com/catalyst-team/catalyst/pull/841))
- `global_*` counters in `Runner` ([#858](https://github.com/catalyst-team/catalyst/pull/858))
- EarlyStoppingCallback considers first epoch as bad ([#854](https://github.com/catalyst-team/catalyst/issues/854))
- annoying numpy warning ([#860](https://github.com/catalyst-team/catalyst/pull/860))
- `PeriodicLoaderCallback` overwrites best state ([#867](https://github.com/catalyst-team/catalyst/pull/867))
- `OneCycleLRWithWarmup` ([#851](https://github.com/catalyst-team/catalyst/issues/851))

## [20.06] - 2020-06-04

### Added

- `Mergify` ([#831](https://github.com/catalyst-team/catalyst/pull/831))
- `PerplexityMetricCallback` ([#819](https://github.com/catalyst-team/catalyst/pull/819))
- `PeriodicLoaderRunnerCallback` ([#818](https://github.com/catalyst-team/catalyst/pull/818))

### Changed

- docs structure were updated during ([#822](https://github.com/catalyst-team/catalyst/pull/822))
- `utils.process_components` moved from `utils.distributed` to `utils.components` ([#822](https://github.com/catalyst-team/catalyst/pull/822))
- `catalyst.core.state.State` merged to `catalyst.core.runner._Runner` ([#823](https://github.com/catalyst-team/catalyst/pull/823)) (backward compatibility included)
    - `catalyst.core.callback.Callback` now works directly with `catalyst.core.runner._Runner`
    - `state_kwargs` renamed to `stage_kwargs`

### Removed

- 

### Fixed

- added missed dashes in docker perfixes ([#828](https://github.com/catalyst-team/catalyst/issues/828))
- handle empty loader in Runner ([#873](https://github.com/catalyst-team/catalyst/pull/873))


## [20.05.1] - 2020-05-23

### Added

- Circle loss implementation ([#802](https://github.com/catalyst-team/catalyst/pull/802))
- BatchBalanceSampler for metric learning and classification ([#806](https://github.com/catalyst-team/catalyst/pull/806))
- `CheckpointCallback`: new argument `load_on_stage_start` which accepts `str` and `Dict[str, str]` ([#797](https://github.com/catalyst-team/catalyst/pull/797))
- LanguageModelingDataset to catalyst\[nlp\] ([#808](https://github.com/catalyst-team/catalyst/pull/808))
- Extra counters for batches, loaders and epochs ([#809](https://github.com/catalyst-team/catalyst/pull/809))
- `TracerCallback` ([#789](https://github.com/catalyst-team/catalyst/pull/789))

### Changed

- `CheckpointCallback`: additional logic for argument `load_on_stage_end` - accepts `str` and `Dict[str, str]` ([#797](https://github.com/catalyst-team/catalyst/pull/797))
- counters names for batches, loaders and epochs ([#809](https://github.com/catalyst-team/catalyst/pull/809))
- `utils.trace_model`: changed logic - `runner` argument was changed to `predict_fn` ([#789](https://github.com/catalyst-team/catalyst/pull/789))
- redesigned `contrib.data` and `contrib.datasets` ([#820](https://github.com/catalyst-team/catalyst/pull/820))
- `catalyst.utils.meters` moved to `catalyst.tools` ([#820](https://github.com/catalyst-team/catalyst/pull/820))
- `catalyst.contrib.utils.tools.tensorboard` moved to `catalyst.contrib.tools` ([#820](https://github.com/catalyst-team/catalyst/pull/820))

### Removed

- 

### Fixed

- device selection fix for [#798](https://github.com/catalyst-team/catalyst/issues/798) ([#815](https://github.com/catalyst-team/catalyst/pull/815))
- batch size counting fix for [#799](https://github.com/catalyst-team/catalyst/issues/799) and [#755](https://github.com/catalyst-team/catalyst/issues/755) issues ([#809](https://github.com/catalyst-team/catalyst/pull/809))


## [20.05] - 2020-05-07

### Added

- Added new docs and minimal examples ([#747](https://github.com/catalyst-team/catalyst/pull/747))
- Added experiment to registry ([#746](https://github.com/catalyst-team/catalyst/pull/746))
- Added examples with extra metrics ([#750](https://github.com/catalyst-team/catalyst/pull/750))
- Added VAE example ([#752](https://github.com/catalyst-team/catalyst/pull/752))
- Added gradient tracking ([#679](https://github.com/catalyst-team/catalyst/pull/679)
- Added dependabot ([#771](https://github.com/catalyst-team/catalyst/pull/771))
- Added new test for Config API ([#768](https://github.com/catalyst-team/catalyst/pull/768))
- Added Visdom logger ([#769](https://github.com/catalyst-team/catalyst/pull/769))
- Added new github actions and templates ([#777](https://github.com/catalyst-team/catalyst/pull/777))
- Added `save_n_best=0` support for CheckpointCallback ([#784](https://github.com/catalyst-team/catalyst/pull/784))
- Added new contrib modules for CV ([#793](https://github.com/catalyst-team/catalyst/pull/793))
- Added new github actions CI ([#791](https://github.com/catalyst-team/catalyst/pull/791))

### Changed

- Changed `Alchemy` dependency (from `alchemy-catalyst` to `alchemy`) ([#748](https://github.com/catalyst-team/catalyst/pull/748))
- Changed warnings logic ([#719](https://github.com/catalyst-team/catalyst/pull/719))
- Github actions CI was updated ([#754](https://github.com/catalyst-team/catalyst/pull/754))
- Changed default `num_epochs` to 1 for `State` ([#756](https://github.com/catalyst-team/catalyst/pull/756))
- Changed `state.batch_in`/`state.batch_out` to `state.input`/`state.output` ([#763](https://github.com/catalyst-team/catalyst/pull/763))
- Moved `torchvision` dependency from `catalyst` to `catalyst[cv]` ([#738](https://github.com/catalyst-team/catalyst/pull/738)))

### Removed

- GanRunner removed to Catalyst.GAN ([#760](https://github.com/catalyst-team/catalyst/pull/760))
- `monitoring_params` were removed ([#760](https://github.com/catalyst-team/catalyst/pull/760))

### Fixed

- Fixed docker dependencies ([$753](https://github.com/catalyst-team/catalyst/pull/753))
- Fixed `text2embeddding` script ([#722](https://github.com/catalyst-team/catalyst/pull/722))
- Fixed `utils/sys` exception ([#762](https://github.com/catalyst-team/catalyst/pull/762))
- Returned `detach` method ([#766](https://github.com/catalyst-team/catalyst/pull/766))
- Fixed timer division by zero ([#749](https://github.com/catalyst-team/catalyst/pull/749))
- Fixed minimal torch version ([#775](https://github.com/catalyst-team/catalyst/pull/775))
- Fixed segmentation tutorial ([#778](https://github.com/catalyst-team/catalyst/pull/778))
- Fixed Dockerfile dependency ([#780](https://github.com/catalyst-team/catalyst/pull/780))


## [20.04] - 2020-04-06

### Added

- 

### Changed

- 

### Removed

-

### Fixed

- 


## [YY.MM.R] - YYYY-MM-DD

### Added

- 

### Changed

- 

### Removed

- 

### Fixed

- 
