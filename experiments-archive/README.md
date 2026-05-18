# Archived Experiment Code

This folder contains code from older experiments that is not part of the
current active pipeline. It is kept here for clarity and historical context, so
previous model/build attempts can still be inspect without keeping the main
`src/` tree noisy.

Running files directly from `ignore/src/` will probably not work. Many scripts
assume imports such as `from utils...` or `from finn_build...`, which are
resolved relative to the main `src/` tree. From inside `ignore/src/`, those
import paths may point to the wrong place or fail entirely.

This folder is therefore an archive for context, not an alternate source tree. If you want to revive one of them, move it back to the corresponding path
under `src/`. In most cases, no refactoring should be needed after moving the file back to
its original location.

## Contents

`ignore/src/` contains older training, QAT, export, evaluation, and FINN build
scripts for experiments such as:

- CustomNet
- MobileNetV1
- ResNet18
- older non-trim `test_resnet` flows
- checkpoint evaluation helpers

