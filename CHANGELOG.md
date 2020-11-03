# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).


## [YY.MM.R] - YYYY-MM-DD

### Added

- extra functions for classification metrics ([#966](https://github.com/catalyst-team/catalyst/pull/966))
- `OneOf` and `OneOfV2` batch transforms ([#951](https://github.com/catalyst-team/catalyst/pull/951))
- ``precision_recall_fbeta_support`` metric ([#971](https://github.com/catalyst-team/catalyst/pull/971))

### Changed

- update Catalyst version to `20.10.1` for tutorials ([#967](https://github.com/catalyst-team/catalyst/pull/967))
- added link to dl-course ([#967](https://github.com/catalyst-team/catalyst/pull/967))

### Removed

- several deprecated tutorials ([#967](https://github.com/catalyst-team/catalyst/pull/967))

### Fixed

- `BatchTransformCallback` - add `nn.Module` transforms support ([#951](https://github.com/catalyst-team/catalyst/pull/951))
- moved to `contiguous` view for accuracy computation ([#982](https://github.com/catalyst-team/catalyst/pull/982))


## [20.10.1] - 2020-10-15

### Added

- MRR metrics calculation ([#886](https://github.com/catalyst-team/catalyst/pull/886))
- docs for MetricCallbacks ([#947](https://github.com/catalyst-team/catalyst/pull/947)) 
- SoftMax, CosFace, ArcFace layers to contrib ([#939](https://github.com/catalyst-team/catalyst/pull/939))
- ArcMargin layer to contrib ([#957](https://github.com/catalyst-team/catalyst/pull/957))
- AdaCos to contrib ([#958](https://github.com/catalyst-team/catalyst/pull/958))
- Manual SWA to utils ([#945](https://github.com/catalyst-team/catalyst/pull/945))

### Changed

- fixed path to `CHANGELOG.md` file and add information about unit test to `PULL_REQUEST_TEMPLATE.md` ([#955])(https://github.com/catalyst-team/catalyst/pull/955)
- `catalyst-dl tune` config specification - now optuna params are grouped under `study_params` ([#947](https://github.com/catalyst-team/catalyst/pull/947))
- `IRunner._prepare_for_stage` logic moved to `IStageBasedRunner.prepare_for_stage` ([#947](https://github.com/catalyst-team/catalyst/pull/947))
    - now we create components in the following order: datasets/loaders, model, criterion, optimizer, scheduler, callbacks
- `MnistMLDataset` and `MnistQGDataset` data split logic - now targets of the datasets are disjoint ([#949](https://github.com/catalyst-team/catalyst/pull/949))
- architecture redesign ([#953](https://github.com/catalyst-team/catalyst/pull/953))
    - experiments, runners, callbacks grouped by primitives under `catalyst.experiments`/`catalyst.runners`/`catalyst.callbacks` respectively
    - settings and typing moved from `catalyst.tools.*` to `catalyst.*`
    - utils moved from `catalyst.*.utils` to `catalyst.utils`
- swa moved to `catalyst.utils` ([#963](https://github.com/catalyst-team/catalyst/pull/963))

### Removed

- 

### Fixed

- `AMPOptimizerCallback` - fix grad clip fn support ([#948](https://github.com/catalyst-team/catalyst/pull/948))
- removed deprecated docs types ([#947](https://github.com/catalyst-team/catalyst/pull/947)) ([#952](https://github.com/catalyst-team/catalyst/pull/952))
- docs for a few files ([#952](https://github.com/catalyst-team/catalyst/pull/952))
- extra backward compatibility fixes ([#963](https://github.com/catalyst-team/catalyst/pull/963))


## [20.09.1] - 2020-09-25

### Added

- Runner registry support for Config API ([#936](https://github.com/catalyst-team/catalyst/pull/936))

- `catalyst-dl tune` command - Optuna with Config API integration for AutoML hyperparameters optimization ([#937](https://github.com/catalyst-team/catalyst/pull/937))
- `OptunaPruningCallback` alias for `OptunaCallback` ([#937](https://github.com/catalyst-team/catalyst/pull/937))
- AdamP and SGDP to `catalyst.contrib.nn.criterion` ([#942](https://github.com/catalyst-team/catalyst/pull/942))

### Changed

- Config API components preparation logic moved to ``utils.prepare_config_api_components`` ([#936](https://github.com/catalyst-team/catalyst/pull/936))

### Removed

- 

### Fixed

- Logging double logging :) ([#936](https://github.com/catalyst-team/catalyst/pull/936))

- CMCCallback ([#941](https://github.com/catalyst-team/catalyst/pull/941))

## [20.09] - 2020-09-07

### Added

- `MovieLens dataset` loader ([#903](https://github.com/catalyst-team/catalyst/pull/903))
- `force` and `bert-level` keywords to `catalyst-data text2embedding` ([#917](https://github.com/catalyst-team/catalyst/pull/917))
- `OptunaCallback` to `catalyst.contrib` ([#915](https://github.com/catalyst-team/catalyst/pull/915))
- `DynamicQuantizationCallback` and `catalyst-dl quantize` script for fast quantization of your model ([#890](https://github.com/catalyst-team/catalyst/pull/915))
- Multi-scheduler support for multi-optimizer case ([#923](https://github.com/catalyst-team/catalyst/pull/923))
- Native mixed-precision training support ([#740](https://github.com/catalyst-team/catalyst/issues/740))
- `OptiomizerCallback` - flag `use_fast_zero_grad` for faster (and hacky) version of `optimizer.zero_grad()` ([#927](https://github.com/catalyst-team/catalyst/pull/927))
- `IOptiomizerCallback`, `ISchedulerCallback`, `ICheckpointCallback`, `ILoggerCallback` as core abstractions for Callbacks ([#933](https://github.com/catalyst-team/catalyst/pull/933))
- flag `USE_AMP` for PyTorch AMP usage ([#933](https://github.com/catalyst-team/catalyst/pull/933))

### Changed

- Pruning moved to `catalyst.dl` ([#933](https://github.com/catalyst-team/catalyst/pull/933))
- default `USE_APEX` changed to 0 ([#933](https://github.com/catalyst-team/catalyst/pull/933))

### Removed

- 

### Fixed

- autoresume option for Config API ([#907](https://github.com/catalyst-team/catalyst/pull/907))
- a few issues with TF projector ([#917](https://github.com/catalyst-team/catalyst/pull/917))
- batch sampler speed issue ([#921](https://github.com/catalyst-team/catalyst/pull/921)) 
- add apex key-value optimizer support ([#924](https://github.com/catalyst-team/catalyst/pull/924))
- runtime warning for PyTorch 1.6 ([920](https://github.com/catalyst-team/catalyst/pull/920))
- Apex synbn usage ([920](https://github.com/catalyst-team/catalyst/pull/920))
- Catalyst dependency on system git ([922](https://github.com/catalyst-team/catalyst/pull/922))


## [20.08] - 2020-08-09

### Added
- `CMCScoreCallback` ([#880](https://github.com/catalyst-team/catalyst/pull/880))
- kornia augmentations `BatchTransformCallback` ([#862](https://github.com/catalyst-team/catalyst/issues/862))
- `average_precision` and `mean_average_precision` metrics ([#883](https://github.com/catalyst-team/catalyst/pull/883))
- `MultiLabelAccuracyCallback`, `AveragePrecisionCallback` and `MeanAveragePrecisionCallback` callbacks ([#883](https://github.com/catalyst-team/catalyst/pull/883))
- minimal examples for multi-class and milti-label classification ([#883](https://github.com/catalyst-team/catalyst/pull/883))
- experimental TPU support ([#893](https://github.com/catalyst-team/catalyst/pull/893))
- add `Imagenette`, `Imagewoof`, and `Imagewang` datasets ([#902](https://github.com/catalyst-team/catalyst/pull/902))
- `IMetricCallback`, `IBatchMetricCallback`, `ILoaderMetricCallback`, `BatchMetricCallback`, `LoaderMetricCallback` abstractions ([#897](https://github.com/catalyst-team/catalyst/pull/897))
- `HardClusterSampler` inbatch sampler ([#888](https://github.com/catalyst-team/catalyst/pull/888))

### Changed

- all registries merged to one `catalyst.registry` ([#883](https://github.com/catalyst-team/catalyst/pull/883))
- `mean_average_precision` logic merged with `average_precision` ([#897](https://github.com/catalyst-team/catalyst/pull/897))
- all imports moved to absolute ([#905](https://github.com/catalyst-team/catalyst/pull/905))
- `catalyst.contrib.data` merged to `catalyst.data` ([#905](https://github.com/catalyst-team/catalyst/pull/905))
- {breaking} Catalyst transform `ToTensor` was renamed to `ImageToTensor` ([#905](https://github.com/catalyst-team/catalyst/pull/905))
- `TracerCallback` moved to `catalyst.dl` ([#905](https://github.com/catalyst-team/catalyst/pull/905))
- `ControlFlowCallback`, `PeriodicLoaderCallback` moved to `catalyst.core` ([#905](https://github.com/catalyst-team/catalyst/pull/905))

### Removed

- `average_accuracy` and `mean_average_accuracy` metrics ([#883](https://github.com/catalyst-team/catalyst/pull/883))
- MultiMetricCallback abstraction ([#897](https://github.com/catalyst-team/catalyst/pull/897))

### Fixed

- `utils.tokenize_text` typo with punctuation ([#880](https://github.com/catalyst-team/catalyst/pull/880))
- `ControlFlowCallback` logic ([#892](https://github.com/catalyst-team/catalyst/pull/892))
- docs ([#897](https://github.com/catalyst-team/catalyst/pull/897))


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
