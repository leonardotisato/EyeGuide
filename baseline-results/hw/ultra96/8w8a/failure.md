# Ultra96 8w8a hardware failure

## Summary

The `test_resnet` `8w8a` FINN build targeting `Ultra96 (xczu3eg)` failed during `step_synthesize_bitfile`.

Vivado stopped before placement because the design does not fit the device resources.

## Failure stage

- FINN step: `step_synthesize_bitfile`
- Vivado stage: pre-placement DRC before `place_design`
- Error class: `DRC UTLZ-1`

## What failed

- BRAM18-equivalent demand: `770 / 432`
  - over by `338`
- RAMB36 demand: `329 / 216`
  - over by `113`
- CLB LUT demand after optimization: `71817 / 70560`
  - over by `1257`

The dominant blocker is BRAM over-utilization. LUTs are also slightly over capacity.

## What did not fail

- CLB Registers: `78393 / 141120` (`55.55%`)
- DSPs: `15 / 360` (`4.17%`)

Timing was not the failure mode. Vivado did not reach actual placement because resource DRC failed first.

## Key Vivado messages

- `RAMB18 and RAMB36/FIFO over-utilized in Top Level Design`
- `This design requires 770 of such cell types but only 432 compatible sites are available`
- `RAMB36/FIFO over-utilized in Top Level Design`
- `This design requires 329 of such cell types but only 216 compatible sites are available`
- `Placer not run`

## Recorded conclusion

`Ultra96` `8w8a` is a negative result: the overlay does not fit on the target device, primarily because of BRAM pressure, with a smaller concurrent LUT overflow.

