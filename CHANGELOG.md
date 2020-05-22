# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [20.06] - YYYY-MM-DD

### Added


- Added Circle loss implementation ([#802](https://github.com/catalyst-team/catalyst/pull/802))
- Added BatchBalanceSampler for metric learning and classification ([#806](https://github.com/catalyst-team/catalyst/pull/806))
- `CheckpointCallback`: new argument `load_on_stage_start` which accepts `str` and `Dict[str, str]` ([#797](https://github.com/
- Add LanguageModelingDataset to catalyst\[nlp\] ([#808](https://github.com/catalyst-team/catalyst/pull/808))


### Changed

- `CheckpointCallback`: additional logic for argument `load_on_stage_end` - accepts `str` and `Dict[str, str]` ([#797](https://github.com/catalyst-team/catalyst/pull/797))

### Removed

- 

### Fixed

- device selection fix for [#798](https://github.com/catalyst-team/catalyst/issues/798) ([#815](https://github.com/catalyst-team/catalyst/pull/815))

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
