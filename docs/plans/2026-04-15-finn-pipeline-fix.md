# FINN Pipeline Fix — test_resnet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `custom_steps_test_resnet.py` so the full FINN dataflow build pipeline completes without a partition cycle for the 8w8a QAT test_resnet model.

**Architecture:** Step-by-step inspection of existing intermediate ONNX files using Python + onnx library (no Docker needed for inspection), followed by targeted fixes applied to `custom_steps_test_resnet.py`. FINN source is also available locally under `ignore/finn-src/` for reference. Re-run inside Docker for verification.

**Tech Stack:** FINN v0.10.1, Brevitas, QONNX, Python `onnx` library, `xilinx/finn:v0.10.1-6-g8ac41e46` Docker image.

---

## Context

The build fails at `step_create_dataflow_partition` because non-HW nodes are **sandwiched inside the residual connections**, creating a cycle in the parent graph when FINN tries to isolate the dataflow partition.

From the April 15 build log, after `step_test_resnet_to_hw` the following remain non-HW:

```
{'Transpose': 8, 'MaxPoolNHWC': 1, 'Add': 2, 'Mul': 4,
 'AveragePool': 3, 'Trunc': 3, 'MatMul': 3, 'MultiThreshold': 3, 'TopK': 1}
```

In topological order (from the WARNING lines):
```
Transpose_0, Transpose_1               ← input edge (probably OK)
MaxPoolNHWC_0                          ← stem MaxPool k=3,s=2 — BLOCKER
Add_0                                  ← layer1 residual add — BLOCKER
Transpose_2→Mul_0→AveragePool_0→Trunc_0→Transpose_3→MatMul_0→MultiThreshold_0  ← layer2 downsample
Transpose_4→Mul_1→AveragePool_1→Trunc_1→Transpose_5→MatMul_1→MultiThreshold_1  ← layer3 downsample
Transpose_6→Mul_2→AveragePool_2→Trunc_2→Transpose_7→MatMul_2→MultiThreshold_2  ← layer4 downsample
Mul_3→Add_1                            ← unknown residual — BLOCKER
TopK_0                                 ← output edge (probably OK)
```

The residual Add nodes with non-HW inputs create the partition cycle. Every non-HW node inside a residual path must be converted. The overall fix categories are:

- **MaxPoolNHWC** (k=3, s=2): `InferStreamingMaxPool` requires k=s and rejects this. Fix strategy TBD (see Task 3).
- **AveragePool+Trunc ×3**: TruncAvgPool2d downsample nodes — `InferPool` is missing from the pipeline.
- **Mul ×3 before AveragePool**: Scale Muls from preceding QuantReLU that couldn't be moved past AvgPool.
- **MatMul+MultiThreshold ×3 after AveragePool**: Downsample Conv1x1+BN nodes blocked by unknown layout from non-HW AvgPool.
- **Add_0, Add_1**: Cannot be converted while any input branch has non-HW nodes.
- **Mul_3, TopK_0**: Investigate (may resolve automatically once other nodes fix).

---

## Phase 1 — Inspect Existing Intermediate ONNX Files

The intermediate ONNX files from the April 15 run (steps 1–6) already exist in
`build_finn_test_resnet/intermediate_models/`. Inspect them with the `onnx` library
to understand the exact graph structure before writing any fix.

---

### Task 1: Inspect the after-lower graph

**Files:**
- Read: `build_finn_test_resnet/intermediate_models/step_test_resnet_lower.onnx`

This reveals the graph state AFTER `LowerConvsToMatMul`, `MakeMaxPoolNHWC`, and transpose absorptions —
right before HW conversion attempts. We need to see:
- Exact node names/types around MaxPoolNHWC
- The AveragePool+Trunc pattern from TruncAvgPool2d
- Whether Muls appear before or after AveragePool
- Transpose positions around AvgPool and MaxPool

- [ ] **Step 1: Write and run the inspection script (outside Docker, needs `pip install onnx`)**

```python
# Run from project root: python docs/plans/inspect_lower.py
import onnx
model = onnx.load("build_finn_test_resnet/intermediate_models/step_test_resnet_lower.onnx")
graph = model.graph

# Print all nodes with their op_type, name, inputs, outputs
for i, n in enumerate(graph.node):
    print(f"[{i:03d}] {n.op_type:30s} {n.name:35s}  in={list(n.input)}  out={list(n.output)}")
```

Run: `python docs/plans/inspect_lower.py 2>/dev/null | head -120`

- [ ] **Step 2: Report the output**

Report the full printed node list. Key things to look for:
- Where does `MaxPoolNHWC` appear? What are its input/output tensor names?
- What precedes `MaxPoolNHWC` (Transpose? MultiThreshold?) and what follows it?
- For each `AveragePool`: what are its inputs? Is there a `Mul` immediately before it?
- Does `Trunc` always immediately follow `AveragePool`?
- Are there `Transpose` nodes between `Trunc` and the subsequent `MatMul`?

---

### Task 2: Inspect the after-to_hw graph

**Files:**
- Read: `build_finn_test_resnet/intermediate_models/step_test_resnet_to_hw.onnx`

This is the graph that CAUSES the partition cycle. We need to understand the exact
position of every non-HW node relative to HW nodes.

- [ ] **Step 1: Write and run the inspection script**

```python
# Run from project root: python docs/plans/inspect_to_hw.py
import onnx
model = onnx.load("build_finn_test_resnet/intermediate_models/step_test_resnet_to_hw.onnx")
graph = model.graph

HW_DOMAINS = {
    "finn.custom_op.fpgadataflow",
    "finn.custom_op.fpgadataflow.hlsbackend",
    "finn.custom_op.fpgadataflow.rtlbackend",
}

for i, n in enumerate(graph.node):
    hw = "HW " if n.domain in HW_DOMAINS else "    "
    print(f"[{i:03d}] {hw}{n.op_type:40s} {n.name}")
```

Run: `python docs/plans/inspect_to_hw.py 2>/dev/null`

- [ ] **Step 2: Report the output**

Report the full node list (HW vs non-HW). Key things to identify:
- Which HW nodes flank `MaxPoolNHWC_0`? (Both sides should be HW convolutions.)
- Are the downsample path nodes `Mul→AveragePool→Trunc→MatMul→MultiThreshold` a contiguous non-HW block?
- Which HW nodes flank `Add_0` and `Add_1`? Are both inputs of each Add clearly identified?
- What precedes `TopK_0`? (Is it a HW node or a non-HW node?)
- Are `Transpose_0` and `Transpose_1` at the very start (input edge)?
- Do `Transpose_2–7` appear WITHIN the downsample chains?

---

### Task 3: Check FINN source for MaxPool and InferPool handling

**Files:**
- Read: `ignore/finn-src/src/finn/transformation/fpgadataflow/convert_to_hw_layers.py`
  (sections: `InferStreamingMaxPool`, `InferPool`)

This determines whether `InferPool` can handle `MaxPoolNHWC` (k=3, s=2) OR if we need to
change the architecture.

- [ ] **Step 1: Search the FINN source for InferPool and InferStreamingMaxPool**

```bash
grep -n "class InferPool\|class InferStreamingMaxPool\|MaxPoolNHWC\|op_type.*Max\|op_type.*Pool\|ks == s\|kernel.*stride" \
  ignore/finn-src/src/finn/transformation/fpgadataflow/convert_to_hw_layers.py | head -60
```

- [ ] **Step 2: Read the full bodies of both transforms**

```bash
# Find line numbers first
grep -n "class InferPool\|class InferStreamingMaxPool" \
  ignore/finn-src/src/finn/transformation/fpgadataflow/convert_to_hw_layers.py
```

Then read approximately 80 lines from each class start.

- [ ] **Step 3: Report and decide**

Report what op_types `InferPool` handles:

**If `InferPool` handles `MaxPoolNHWC`:**
→ Add `InferPool()` to `step_test_resnet_to_hw`. No architecture change needed. Continue to Task 4.

**If `InferPool` does NOT handle `MaxPoolNHWC`:**
→ Two sub-options:
  - **3A (preferred)**: Change `quant_test_resnet.py` stem from
    `nn.MaxPool2d(kernel_size=3, stride=2, padding=1)` to `nn.MaxPool2d(kernel_size=2, stride=2, padding=0)`.
    Output size is identical (both produce 56×56 from 112×112 input via same formula).
    Re-export the ONNX without re-doing QAT (MaxPool has no weights; accuracy impact is negligible
    as the QAT-trained model will adapt at the quantizer scale level — verify in Task 9).
  - **3B (fallback)**: Write a custom `ConvertMaxPoolNHWCToHW` transform that calls
    `InferPool` logic on `MaxPoolNHWC` nodes directly.

---

### Task 4: Check FINN source for how TruncAvgPool2d is handled

**Files:**
- Read: `ignore/finn-src/src/finn/transformation/fpgadataflow/convert_to_hw_layers.py`
  (section: `InferPool`, specifically AvgPool+Trunc pattern)

- [ ] **Step 1: Check if InferPool handles AveragePool+Trunc pairs**

```bash
grep -n "Trunc\|AveragePool\|Pool_Batch\|op_type" \
  ignore/finn-src/src/finn/transformation/fpgadataflow/convert_to_hw_layers.py | grep -A2 -B2 "Trunc"
```

- [ ] **Step 2: Check the CustomSmallNet intermediate model for comparison**

The CustomSmallNet build DID work with TruncAvgPool2d (confirmed by CHANGELOG). Inspect its
intermediate to confirm `InferPool` was the fix:

```bash
ls build_finn_custom_net/intermediate_models/ 2>/dev/null || echo "not found"
```

If the custom_net build dir exists, run:
```python
# python docs/plans/inspect_custom_net_to_hw.py
import onnx
model = onnx.load("build_finn_custom_net/intermediate_models/step_custom_net_to_hw.onnx")
HW_DOMAINS = {"finn.custom_op.fpgadataflow", "finn.custom_op.fpgadataflow.hlsbackend", "finn.custom_op.fpgadataflow.rtlbackend"}
for i, n in enumerate(model.graph.node):
    hw = "HW " if n.domain in HW_DOMAINS else "    "
    print(f"[{i:03d}] {hw}{n.op_type}")
```

Confirm that no `AveragePool` or `Trunc` nodes remain non-HW. If confirmed, `InferPool` is the correct fix for these.

- [ ] **Step 3: Report findings**

Report: Does `InferPool` in the FINN source specifically handle `AveragePool`+`Trunc` pairs, or just plain `AveragePool`? Does it need the Trunc to be present/absent?

---

### Task 5: Investigate the pre-AvgPool Mul nodes

**Files:**
- Read: `ignore/finn-src/src/finn/transformation/streamline/reorder.py`
  (section: `MoveMulPastMaxPool`)

The 3 Mul nodes before each AveragePool in the downsample path are scale Muls from the preceding
QuantReLU that couldn't be moved past the AvgPool during streamlining. This needs to be fixed
EITHER in the streamline step (by adding MoveMulPastAvgPool if it exists) OR as an absorption
step before `InferPool` in `step_test_resnet_to_hw`.

- [ ] **Step 1: Check if MoveMulPastAvgPool or a combined version exists**

```bash
grep -n "class MoveMulPast\|AveragePool\|AvgPool" \
  ignore/finn-src/src/finn/transformation/streamline/reorder.py | head -20
```

- [ ] **Step 2: Check if InferPool absorbs the preceding Mul automatically**

Read ~50 lines of the `InferPool` body (from the class start found in Task 3) to see if it walks
backwards past a Mul node to absorb it, or if the input Mul must be gone before InferPool runs.

- [ ] **Step 3: Report findings**

Report one of:
- **Option A**: `MoveMulPastMaxPool` in FINN also handles `AveragePool` → Muls will be moved during streamlining if we add AvgPool to its target list. But this requires a custom fork — not ideal.
- **Option B**: `InferPool` handles AvgPool WITH a preceding Mul → we just need to add `InferPool` and the Mul gets absorbed.
- **Option C**: The Mul needs to be explicitly absorbed before `InferPool` runs → we need `AbsorbMulIntoMultiThreshold` or a custom absorb after `InferPool` converts the AvgPool.

---

### Task 6: Identify Add_0 and Add_1 precisely

**Files:**
- Read: `build_finn_test_resnet/intermediate_models/step_test_resnet_to_hw.onnx`

From Task 2's output, identify the exact inputs to Add_0 and Add_1. This determines:
- Which residual blocks they belong to
- Which input branches are still non-HW
- Whether fixing the other issues (MaxPool, AvgPool) will unblock them automatically

- [ ] **Step 1: Using Task 2's printed output, trace each Add**

For Add_0:
- Find `Add_0` in the node list
- Look up its two inputs (tensor names) → find which nodes produce those tensors
- Write: "Add_0: input A comes from [node/type], input B comes from [node/type]"

For Add_1:
- Same analysis

- [ ] **Step 2: Confirm which residual blocks they belong to**

Given the model structure:
- Layer1: BasicBlock, no downsample (identity = MaxPool output)
- Layer2: BasicBlock, ds = TruncAvgPool2d+Conv1x1
- Layer3: Bottleneck, ds = TruncAvgPool2d+Conv1x1
- Layer4: BasicBlock, ds = TruncAvgPool2d+Conv1x1

Which Add is from which layer? The answer determines which non-HW nodes are their blockers.

**Expected answer:** Add_0 = layer1 residual (blocked by MaxPoolNHWC). Add_1 = one of the
downsample-layer Adds (blocked by AveragePool chain). The other 2 downsample Adds were either
converted or are also in the list (need to confirm with the actual output from Task 2).

---

## Phase 2 — Apply All Fixes

Once Phase 1 is complete and all findings are reported, apply the following fixes.
The exact implementation of each fix may need adjustment based on Phase 1 findings.

---

### Task 7: Fix MaxPool (k=3, s=2) incompatibility

**Files:**
- Possibly modify: `src/utils/quant_test_resnet.py` (if architecture change path chosen)
- Possibly modify: `src/export_test_resnet.py` (to re-export)
- Modify: `src/finn_build/custom_steps_test_resnet.py`

Based on Task 3's decision:

- [ ] **Step 1: Apply the chosen fix**

**If Option 3A (change MaxPool k=3→k=2, re-export):**

In `src/utils/quant_test_resnet.py`, line 220:
```python
# BEFORE
self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
# AFTER
self.stem_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
```
Output shape is identical: both map 112×112 → 56×56.

Then re-export (the QAT checkpoint is not re-trained — this only changes the export topology,
the scale parameters from QAT are preserved):
```bash
# Inside Docker or locally (requires Brevitas)
python src/export_test_resnet.py
```
Expected: `models/test_resnet_8w8a.onnx` overwritten. Numerical check passes.

**If Option 3B (custom transform for MaxPoolNHWC):**

Write a `ConvertMaxPoolNHWCToPool` transform in `custom_steps_test_resnet.py` that converts
`MaxPoolNHWC` nodes to `Pool_Batch` HW nodes directly. (Exact implementation depends on
`Pool_Batch`'s node attributes found in Task 3.)

- [ ] **Step 2: Confirm `MakeMaxPoolNHWC` and `InferStreamingMaxPool` still appear in the right places**

If we change to k=2 (3A): `InferStreamingMaxPool` will now handle it (k=s=2). Keep `MakeMaxPoolNHWC` in lower step and `InferStreamingMaxPool` in to_hw step — no further changes needed for MaxPool.

If 3B: add `ConvertMaxPoolNHWCToPool` in `step_test_resnet_to_hw` after `InferConvInpGen`.

---

### Task 8: Add InferPool to step_test_resnet_to_hw

**Files:**
- Modify: `src/finn_build/custom_steps_test_resnet.py`

Add `InferPool` to convert the `AveragePool`(+`Trunc`) nodes from TruncAvgPool2d in all
three downsample paths. This is the most critical missing transform.

- [ ] **Step 1: Add InferPool to the import**

In `custom_steps_test_resnet.py`, find the import block:
```python
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferAddStreamsLayer,
    InferGlobalAccPoolLayer,
    InferStreamingMaxPool,
    ...
)
```
Add `InferPool` to the list:
```python
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferAddStreamsLayer,
    InferGlobalAccPoolLayer,
    InferPool,                        # ← ADD THIS
    InferStreamingMaxPool,
    ...
)
```

- [ ] **Step 2: Add InferPool() to to_hw_transformations**

In `step_test_resnet_to_hw`, `to_hw_transformations` list, add `InferPool()` after
`AbsorbScalarMulIntoMatMul()` and before `InferStreamingMaxPool()` — same ordering as
`custom_steps_custom_net.py`:

```python
to_hw_transformations = [
    DoubleToSingleFloat(),
    InferDataTypes(),
    SortGraph(),
    InferShapes(),
    MoveTransposePastJoinAdd(),
    AbsorbTransposeIntoMultiThreshold(),
    AbsorbConsecutiveTransposes(),
    InferAddStreamsLayer(),
    InferGlobalAccPoolLayer(),
    AbsorbScalarMulIntoMatMul(),
    InferPool(),                      # ← ADD THIS (converts AvgPool+Trunc from TruncAvgPool2d)
    InferStreamingMaxPool(),
    RoundAndClipThresholds(),
    FixThresholdDataTypes(),
    InferThresholdingLayer(),
    InferConvInpGen(),
    InferQuantizedMatrixVectorActivation(),
    InferDuplicateStreamsLayer(),
    InferChannelwiseLinearLayer(),
    InferLabelSelectLayer(),
    AbsorbConsecutiveTransposes(),
    AbsorbTransposeIntoFlatten(),
    RemoveCNVtoFCFlatten(),
]
```

---

### Task 9: Fix the Mul nodes before AveragePool

**Files:**
- Modify: `src/finn_build/custom_steps_test_resnet.py`

Based on Task 5's findings, apply the correct fix:

- [ ] **Step 1: Apply appropriate fix**

**If InferPool absorbs the preceding Mul automatically** (Option B from Task 5):
→ No additional change needed beyond Task 8.

**If the Mul must be explicitly handled** (Option C):
Add an explicit `AbsorbMulIntoMultiThreshold()` pass AFTER `InferPool()` in `to_hw_transformations`.
This ensures any Muls that survive InferPool conversion get absorbed:

```python
    InferPool(),
    AbsorbMulIntoMultiThreshold(),    # ← Add if Mul nodes persist after InferPool
    InferStreamingMaxPool(),
    ...
```

Alternatively, if `MoveMulPastMaxPool` also handles `AveragePool` (Option A), add an extra
pass in the post-lower streamline section of `step_test_resnet_to_hw`:
```python
    MoveMulPastMaxPool(),             # ← Add here to move Muls past AvgPool before InferPool
    AbsorbMulIntoMultiThreshold(),
    AbsorbMulIntoMultiThreshold(),    # second pass needed if first exposes new patterns
```

---

### Task 10: Add an explicit post-InferPool streamline pass

**Files:**
- Modify: `src/finn_build/custom_steps_test_resnet.py`

After `InferPool` converts the AvgPool nodes and the downstream MatMul+MultiThreshold
become reachable, a second round of streamline/absorb may be needed to clean up any
remaining floating-point Mul/Add artifacts before `InferQuantizedMatrixVectorActivation`
can complete conversion.

- [ ] **Step 1: Add a mini-streamline pass between InferPool and InferConvInpGen**

Between `InferPool` and `RoundAndClipThresholds` in `to_hw_transformations`, add:

```python
    InferPool(),
    # After InferPool, re-absorb any Muls that were stuck before AvgPool and are now exposed
    AbsorbMulIntoMultiThreshold(),
    Absorb1BitMulIntoConv(),
    CollapseRepeatedMul(),
    RoundAndClipThresholds(),
    FixThresholdDataTypes(),
    InferThresholdingLayer(),
    InferConvInpGen(),
    InferStreamingMaxPool(),
    ...
```

---

### Task 11: Fix stale comments in custom_steps_test_resnet.py

**Files:**
- Modify: `src/finn_build/custom_steps_test_resnet.py`

Two module-level comments are wrong:

- [ ] **Step 1: Fix module docstring**

Update the top-level docstring to reflect actual differences:
```python
"""
Differences from ResNet18 pipeline:
  - Stem MaxPool present → MoveMulPastMaxPool, MakeMaxPoolNHWC, InferStreamingMaxPool included
  - TruncAvgPool2d in downsample paths → InferPool required
  - GlobalAvgPool → needs InferGlobalAccPoolLayer + AbsorbScalarMulIntoMatMul
  - Otherwise identical residual handling
"""
```

- [ ] **Step 2: Fix step_test_resnet_lower docstring (line 156)**

```python
# BEFORE
# Same as ResNet18 but without MakeMaxPoolNHWC (no MaxPool).
# AFTER
# Same as ResNet18 but WITH MakeMaxPoolNHWC (stem has MaxPool, unlike ResNet18).
```

- [ ] **Step 3: Fix step_test_resnet_streamline docstring (line 96)**

```python
# BEFORE
# No MoveMulPastMaxPool (no MaxPool in this model).
# AFTER
# MoveMulPastMaxPool included ×2 — needed for stem MaxPool.
```

---

## Phase 3 — Verification

### Task 12: First run — stop after step_test_resnet_to_hw

**Files:**
- Read: build output from `build_finn_test_resnet/build_dataflow.log`

Run the pipeline stopping after `step_test_resnet_to_hw` to check the non-HW count.
All nodes inside residual paths must be zero. Only `Transpose` (input/output edges)
and `TopK` (output edge) are acceptable non-HW nodes.

- [ ] **Step 1: Run (inside Docker)**

```bash
# Inside FINN Docker container, from /workspace/hpps
python src/finn_build/build_test_resnet.py \
  --stop-after step_test_resnet_to_hw \
  --output-dir ./build_finn_test_resnet
```

- [ ] **Step 2: Check the non-HW count from the log**

Look for the line:
```
[after to_hw] Total=N  HW=M  Non-HW=K
Non-HW: {...}
```

Expected acceptable non-HW (edge nodes only):
```
{'Transpose': ≤2, 'TopK': 1}
```
All of `MaxPoolNHWC`, `Add`, `Mul`, `AveragePool`, `Trunc`, `MatMul`, `MultiThreshold`
must be **zero**. Any remaining non-zero entry in those categories indicates a transform
is still missing or in the wrong order.

- [ ] **Step 3: Report and iterate if needed**

If non-HW nodes remain inside the graph:
1. Run the inspection script from Task 2 on the new `step_test_resnet_to_hw.onnx`
2. Identify which node type and its neighbors
3. Check if an additional or reordered transform is needed
4. Apply fix and repeat Task 12

---

### Task 13: Second run — stop after step_create_dataflow_partition

- [ ] **Step 1: Run**

```bash
python src/finn_build/build_test_resnet.py \
  --stop-after step_create_dataflow_partition \
  --output-dir ./build_finn_test_resnet
```

- [ ] **Step 2: Confirm no partition cycle error**

The log should show `step_create_dataflow_partition` completing without error.
Expected log entries:
```
[TIMESTAMP] Running step: step_create_dataflow_partition [7/11]
[TIMESTAMP] Running step: step_specialize_layers [8/11]
```
(i.e., step 8 starts, meaning step 7 completed successfully)

No `Partition cycle` or `AssertionError` in the log.

---

### Task 14: Estimates-only run

- [ ] **Step 1: Run estimates**

```bash
python src/finn_build/build_test_resnet.py \
  --estimates-only \
  --output-dir ./build_finn_test_resnet
```

- [ ] **Step 2: Check estimate report**

Read `build_finn_test_resnet/report/estimate_layer_resources_hls.json`.
Key expected values (8w8a, PE=1, SIMD=1):

| Resource | Expected (rough) | Available (Ultra96) | 
|---|---|---|
| BRAM_18K | ~100–300 | 432 |
| LUT | ~30K–70K | 70K |
| DSP | <100 | 360 |

If BRAM exceeds 432, we need folding adjustments or further architecture reduction.
Report the actual numbers.

---

### Task 15: Update CHANGELOG

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add entry under today's date**

Add under `## 2026-04-15`:
```markdown
### FINN build pipeline — fixes applied (Task 15)
- Added `InferPool` to `step_test_resnet_to_hw` for:
  - `TruncAvgPool2d` nodes (3×) in downsample paths of layer2/3/4
  - [If applicable] Stem MaxPool via `InferPool` (if k=3 was kept)
- [If applicable] Changed stem MaxPool from k=3,s=2,p=1 → k=2,s=2,p=0 for `InferStreamingMaxPool` compatibility; output shape identical (56×56)
- [If applicable] Added Mul absorption pass after InferPool
- Fixed stale docstring comments in `custom_steps_test_resnet.py`
- Partition cycle resolved: all non-HW nodes inside residual paths converted
- Estimate reports generated: [add BRAM/LUT/DSP from Task 14]
```

---

## Notes for Execution

**Inspection tasks (Tasks 1–6):** Run locally using plain Python + `onnx` library.
The intermediate ONNX files from today's run are already available.

**Fix tasks (Tasks 7–11):** Edit files locally, no Docker needed.

**Verification tasks (Tasks 12–14):** Must run inside the FINN Docker container.
Use `! run_finn.sh` or SSH to Dwarf5 (`ssh 10.79.3.94`) where the Docker image is available.

**Iteration:** If Task 12 reveals remaining non-HW nodes, use the inspection script to
identify them and add the appropriate transform before re-running. The pattern is always:
1. Identify the op_type of the remaining non-HW node
2. Find the correct FINN transform in `convert_to_hw_layers.py`
3. Add it to `to_hw_transformations` in the correct position (after layout/dtype inference,
   after its dependencies are converted, before nodes that depend on it)
4. Re-run.
