# Complete available UGP source index

This release contains **981 source/configuration/document files**, including **137 distinct notebook snapshots** and an additional plain `.py` inspection export for each notebook. It covers local project trees, saved RTX source snapshots, loose historical notebooks, and the source members of available UGP packages. Source files are expanded directly in GitHub; no ZIP download is needed to inspect model code.

Byte-identical source variants are mapped to a canonical file where practical. Complete project trees retain their structure and imports even when a module also appears in an older snapshot. The [source manifest](SOURCE_MANIFEST.json) records every input alias, original hash, publication hash, redaction, and original notebook execution/error count.

## Project source trees

- [FeelWM/TartanGround trainer, configs, scripts and tests](../feelwm-tartanground/src/)
- [Force-specialist continuation](../feelwm-force-specialist/src/)
- [V7 joint specialists](../v7-joint-specialists/src/v7_joint_specialists/)
- [V7 leg/body specialists](../v7-legs-body/src/v7legs_body/)
- [V7 feedback refinement](../v7-feedback/src/v7feedback/)
- [V7 timestamped visual history](../v7-visual-history/src/v7_visual_history/)
- [Independent terrain/foothold implementation](../terrain-foothold-world-model/src/)
- [V7 server runner](../v7-v7f/runner/)
- [Additional supporting and audit code](supporting-code/)

## Notebook history

The counts below describe the original saved notebook state. Outputs were removed for publication. A saved execution count does not prove a clean full run; a filename such as FINAL or REAL does not establish completion, correctness, or measured performance. The adjacent `.cells.py` file exposes all code cells; IPython shell/magic lines and prose-only conclusion cells are comments there. The canonical notebook retains the original cell source.

| Notebook | Python inspection export | Saved executed code cells | Saved error outputs |
|---|---|---:|---:|
| [C7R_SPATIAL_CAUSAL_VISUAL_REPAIR_VALIDATION_ONLY--ab2f6752c3.ipynb](notebooks/C7R_SPATIAL_CAUSAL_VISUAL_REPAIR_VALIDATION_ONLY--ab2f6752c3.ipynb) | [Code cells](notebooks/C7R_SPATIAL_CAUSAL_VISUAL_REPAIR_VALIDATION_ONLY--ab2f6752c3.cells.py) | 0 | 0 |
| [C7S_CAUSAL_HIGHPASS_ALIGNMENT_REPAIR_APPEND--682ff3576e.ipynb](notebooks/C7S_CAUSAL_HIGHPASS_ALIGNMENT_REPAIR_APPEND--682ff3576e.ipynb) | [Code cells](notebooks/C7S_CAUSAL_HIGHPASS_ALIGNMENT_REPAIR_APPEND--682ff3576e.cells.py) | 0 | 0 |
| [FeelWM_TartanDrive2_FAST_EndToEnd_Colab--2573d4ea12.ipynb](notebooks/FeelWM_TartanDrive2_FAST_EndToEnd_Colab--2573d4ea12.ipynb) | [Code cells](notebooks/FeelWM_TartanDrive2_FAST_EndToEnd_Colab--2573d4ea12.cells.py) | 0 | 0 |
| [FeelWM_merged_v7_exact_source--71385a92b5.ipynb](notebooks/FeelWM_merged_v7_exact_source--71385a92b5.ipynb) | [Code cells](notebooks/FeelWM_merged_v7_exact_source--71385a92b5.cells.py) | 0 | 0 |
| [Handwritten_FeelWM_Geometry_Strict_Colab--cf02d6286a.ipynb](notebooks/Handwritten_FeelWM_Geometry_Strict_Colab--cf02d6286a.ipynb) | [Code cells](notebooks/Handwritten_FeelWM_Geometry_Strict_Colab--cf02d6286a.cells.py) | 0 | 0 |
| [Prompt2_3_Assistant_Proposed_V10xFeelWM_Strict_Colab--82e018bc87.ipynb](notebooks/Prompt2_3_Assistant_Proposed_V10xFeelWM_Strict_Colab--82e018bc87.ipynb) | [Code cells](notebooks/Prompt2_3_Assistant_Proposed_V10xFeelWM_Strict_Colab--82e018bc87.cells.py) | 0 | 0 |
| [UGP_C7R_T4_RESCUE_100MIN_FINAL--abadcffec6.ipynb](notebooks/UGP_C7R_T4_RESCUE_100MIN_FINAL--abadcffec6.ipynb) | [Code cells](notebooks/UGP_C7R_T4_RESCUE_100MIN_FINAL--abadcffec6.cells.py) | 4 | 0 |
| [UGP_C7R_T4_RESCUE_100MIN_FINAL-3--b32da47fd2.ipynb](notebooks/UGP_C7R_T4_RESCUE_100MIN_FINAL-3--b32da47fd2.ipynb) | [Code cells](notebooks/UGP_C7R_T4_RESCUE_100MIN_FINAL-3--b32da47fd2.cells.py) | 5 | 0 |
| [UGP_C7R_T4_RESCUE_100MIN_FINAL-4--c87cb88641.ipynb](notebooks/UGP_C7R_T4_RESCUE_100MIN_FINAL-4--c87cb88641.ipynb) | [Code cells](notebooks/UGP_C7R_T4_RESCUE_100MIN_FINAL-4--c87cb88641.cells.py) | 6 | 1 |
| [UGP_C7R_T4_RESCUE_100MIN_FINAL_-4--9049d135be.ipynb](notebooks/UGP_C7R_T4_RESCUE_100MIN_FINAL_-4--9049d135be.ipynb) | [Code cells](notebooks/UGP_C7R_T4_RESCUE_100MIN_FINAL_-4--9049d135be.cells.py) | 7 | 1 |
| [Untitled10-3--7cf24f12d4.ipynb](notebooks/Untitled10-3--7cf24f12d4.ipynb) | [Code cells](notebooks/Untitled10-3--7cf24f12d4.cells.py) | 23 | 1 |
| [Untitled2--7f9a69aaac.ipynb](notebooks/Untitled2--7f9a69aaac.ipynb) | [Code cells](notebooks/Untitled2--7f9a69aaac.cells.py) | 5 | 1 |
| [Untitled2-1--8208d5314a.ipynb](notebooks/Untitled2-1--8208d5314a.ipynb) | [Code cells](notebooks/Untitled2-1--8208d5314a.cells.py) | 23 | 0 |
| [Untitled2-2--ed6113cdeb.ipynb](notebooks/Untitled2-2--ed6113cdeb.ipynb) | [Code cells](notebooks/Untitled2-2--ed6113cdeb.cells.py) | 30 | 0 |
| [Untitled2-3--1ab8376769.ipynb](notebooks/Untitled2-3--1ab8376769.ipynb) | [Code cells](notebooks/Untitled2-3--1ab8376769.cells.py) | 39 | 0 |
| [V10R_COMPILED_CLEAN_G0B_BASELINE--f6a06e9cb3.ipynb](notebooks/V10R_COMPILED_CLEAN_G0B_BASELINE--f6a06e9cb3.ipynb) | [Code cells](notebooks/V10R_COMPILED_CLEAN_G0B_BASELINE--f6a06e9cb3.cells.py) | 0 | 0 |
| [V10R_Causal_Contact_Patch_World_Model--93f634caf5.ipynb](notebooks/V10R_Causal_Contact_Patch_World_Model--93f634caf5.ipynb) | [Code cells](notebooks/V10R_Causal_Contact_Patch_World_Model--93f634caf5.cells.py) | 0 | 0 |
| [V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-2--11ca15417d.ipynb](notebooks/V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-2--11ca15417d.ipynb) | [Code cells](notebooks/V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-2--11ca15417d.cells.py) | 17 | 2 |
| [V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-3--04c7a244e0.ipynb](notebooks/V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-3--04c7a244e0.ipynb) | [Code cells](notebooks/V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-3--04c7a244e0.cells.py) | 22 | 3 |
| [V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-4--756b0496fe.ipynb](notebooks/V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-4--756b0496fe.ipynb) | [Code cells](notebooks/V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-4--756b0496fe.cells.py) | 25 | 3 |
| [V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-5--76924e728e.ipynb](notebooks/V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-5--76924e728e.ipynb) | [Code cells](notebooks/V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-5--76924e728e.cells.py) | 29 | 3 |
| [V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-6--80e6e46ac9.ipynb](notebooks/V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-6--80e6e46ac9.ipynb) | [Code cells](notebooks/V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-6--80e6e46ac9.cells.py) | 33 | 3 |
| [V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-7--e1acc6a9e3.ipynb](notebooks/V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-7--e1acc6a9e3.ipynb) | [Code cells](notebooks/V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-7--e1acc6a9e3.cells.py) | 39 | 4 |
| [V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-7-2--c697f39180.ipynb](notebooks/V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-7-2--c697f39180.ipynb) | [Code cells](notebooks/V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-7-2--c697f39180.cells.py) | 40 | 4 |
| [V10R_FINAL_END_TO_END_COLAB--ce1b0d14de.ipynb](notebooks/V10R_FINAL_END_TO_END_COLAB--ce1b0d14de.ipynb) | [Code cells](notebooks/V10R_FINAL_END_TO_END_COLAB--ce1b0d14de.cells.py) | 0 | 0 |
| [V10R_FINAL_END_TO_END_COLAB-1--f900ec66b7.ipynb](notebooks/V10R_FINAL_END_TO_END_COLAB-1--f900ec66b7.ipynb) | [Code cells](notebooks/V10R_FINAL_END_TO_END_COLAB-1--f900ec66b7.cells.py) | 49 | 5 |
| [V10R_FINAL_END_TO_END_COLAB-2--9c5242de5d.ipynb](notebooks/V10R_FINAL_END_TO_END_COLAB-2--9c5242de5d.ipynb) | [Code cells](notebooks/V10R_FINAL_END_TO_END_COLAB-2--9c5242de5d.cells.py) | 51 | 0 |
| [V10R_REAL_END_TO_END_TRAINING_AND_RESULTS--e3286e1308.ipynb](notebooks/V10R_REAL_END_TO_END_TRAINING_AND_RESULTS--e3286e1308.ipynb) | [Code cells](notebooks/V10R_REAL_END_TO_END_TRAINING_AND_RESULTS--e3286e1308.cells.py) | 0 | 0 |
| [V10R_REAL_END_TO_END_TRAINING_AND_RESULTS-1--cb946324bc.ipynb](notebooks/V10R_REAL_END_TO_END_TRAINING_AND_RESULTS-1--cb946324bc.ipynb) | [Code cells](notebooks/V10R_REAL_END_TO_END_TRAINING_AND_RESULTS-1--cb946324bc.cells.py) | 0 | 0 |
| [V10R_STAGE_C_RGBD_INCREMENTAL_NO_VLM_NO_PLANNER--5384967e57.ipynb](notebooks/V10R_STAGE_C_RGBD_INCREMENTAL_NO_VLM_NO_PLANNER--5384967e57.ipynb) | [Code cells](notebooks/V10R_STAGE_C_RGBD_INCREMENTAL_NO_VLM_NO_PLANNER--5384967e57.cells.py) | 39 | 0 |
| [V10R_TONIGHT_RESPONSE_WORLD_MODEL1_1_1--5518d2cdb3.ipynb](notebooks/V10R_TONIGHT_RESPONSE_WORLD_MODEL1_1_1--5518d2cdb3.ipynb) | [Code cells](notebooks/V10R_TONIGHT_RESPONSE_WORLD_MODEL1_1_1--5518d2cdb3.cells.py) | 0 | 0 |
| [V10R_TONIGHT_RESPONSE_WORLD_MODEL1_1_1-1--848826f7d3.ipynb](notebooks/V10R_TONIGHT_RESPONSE_WORLD_MODEL1_1_1-1--848826f7d3.ipynb) | [Code cells](notebooks/V10R_TONIGHT_RESPONSE_WORLD_MODEL1_1_1-1--848826f7d3.cells.py) | 39 | 0 |
| [V10R_TONIGHT_RESPONSE_WORLD_MODEL1_1_2-1--901deaadc7.ipynb](notebooks/V10R_TONIGHT_RESPONSE_WORLD_MODEL1_1_2-1--901deaadc7.ipynb) | [Code cells](notebooks/V10R_TONIGHT_RESPONSE_WORLD_MODEL1_1_2-1--901deaadc7.cells.py) | 43 | 3 |
| [V10_FeelWM_FAST_20_30min_EndToEnd_Colab--ddf0cba06f.ipynb](notebooks/V10_FeelWM_FAST_20_30min_EndToEnd_Colab--ddf0cba06f.ipynb) | [Code cells](notebooks/V10_FeelWM_FAST_20_30min_EndToEnd_Colab--ddf0cba06f.cells.py) | 0 | 0 |
| [V10_FeelWM_FROZEN_EXACT_FAST_Colab--0ac65f8f27.ipynb](notebooks/V10_FeelWM_FROZEN_EXACT_FAST_Colab--0ac65f8f27.ipynb) | [Code cells](notebooks/V10_FeelWM_FROZEN_EXACT_FAST_Colab--0ac65f8f27.cells.py) | 0 | 0 |
| [V10_GEOMETRY_COUPLED_FORCE_END_TO_END--e971a2eff7.ipynb](notebooks/V10_GEOMETRY_COUPLED_FORCE_END_TO_END--e971a2eff7.ipynb) | [Code cells](notebooks/V10_GEOMETRY_COUPLED_FORCE_END_TO_END--e971a2eff7.cells.py) | 5 | 1 |
| [V10_GEOMETRY_COUPLED_FORCE_END_TO_END-2--3dca7a728e.ipynb](notebooks/V10_GEOMETRY_COUPLED_FORCE_END_TO_END-2--3dca7a728e.ipynb) | [Code cells](notebooks/V10_GEOMETRY_COUPLED_FORCE_END_TO_END-2--3dca7a728e.cells.py) | 26 | 2 |
| [V10xFeelWM_TartanDrive2_ShortRun_Colab--e843f4d763.ipynb](notebooks/V10xFeelWM_TartanDrive2_ShortRun_Colab--e843f4d763.ipynb) | [Code cells](notebooks/V10xFeelWM_TartanDrive2_ShortRun_Colab--e843f4d763.cells.py) | 0 | 0 |
| [V10xFeelWM_TartanDrive2_ShortRun_Colab_-3--7fbf18c298.ipynb](notebooks/V10xFeelWM_TartanDrive2_ShortRun_Colab_-3--7fbf18c298.ipynb) | [Code cells](notebooks/V10xFeelWM_TartanDrive2_ShortRun_Colab_-3--7fbf18c298.cells.py) | 0 | 0 |
| [V10xFeelWM_TartanDrive2_ShortRun_Colab_-3-2--590318f9b0.ipynb](notebooks/V10xFeelWM_TartanDrive2_ShortRun_Colab_-3-2--590318f9b0.ipynb) | [Code cells](notebooks/V10xFeelWM_TartanDrive2_ShortRun_Colab_-3-2--590318f9b0.cells.py) | 6 | 0 |
| [V7F_2xRTX4080_AUTO_DISCOVER_7_TRUE_STAIRS_FINAL_AUDITED--290881cf41.ipynb](notebooks/V7F_2xRTX4080_AUTO_DISCOVER_7_TRUE_STAIRS_FINAL_AUDITED--290881cf41.ipynb) | [Code cells](notebooks/V7F_2xRTX4080_AUTO_DISCOVER_7_TRUE_STAIRS_FINAL_AUDITED--290881cf41.cells.py) | 0 | 0 |
| [V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT--3374940840.ipynb](notebooks/V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT--3374940840.ipynb) | [Code cells](notebooks/V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT--3374940840.cells.py) | 0 | 0 |
| [V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_CHECKPOINTED_FINAL--38bab346d2.ipynb](notebooks/V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_CHECKPOINTED_FINAL--38bab346d2.ipynb) | [Code cells](notebooks/V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_CHECKPOINTED_FINAL--38bab346d2.cells.py) | 1 | 1 |
| [V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_CHECKPOINTED_FINAL--87adf168cf.ipynb](notebooks/V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_CHECKPOINTED_FINAL--87adf168cf.ipynb) | [Code cells](notebooks/V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_CHECKPOINTED_FINAL--87adf168cf.cells.py) | 3 | 1 |
| [V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_FINAL_AUDITED--af2534a99e.ipynb](notebooks/V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_FINAL_AUDITED--af2534a99e.ipynb) | [Code cells](notebooks/V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_FINAL_AUDITED--af2534a99e.cells.py) | 0 | 0 |
| [V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_SHORT_FOOTHOLD_MEMORY--03b4d808e6.ipynb](notebooks/V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_SHORT_FOOTHOLD_MEMORY--03b4d808e6.ipynb) | [Code cells](notebooks/V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_SHORT_FOOTHOLD_MEMORY--03b4d808e6.cells.py) | 0 | 0 |
| [V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_SHORT_FOOTHOLD_MEMORY--a837ae4272.ipynb](notebooks/V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_SHORT_FOOTHOLD_MEMORY--a837ae4272.ipynb) | [Code cells](notebooks/V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_SHORT_FOOTHOLD_MEMORY--a837ae4272.cells.py) | 0 | 0 |
| [V7F_2xRTX4080_FINAL_OVERNIGHT_RESUMABLE--75f2856f36.ipynb](notebooks/V7F_2xRTX4080_FINAL_OVERNIGHT_RESUMABLE--75f2856f36.ipynb) | [Code cells](notebooks/V7F_2xRTX4080_FINAL_OVERNIGHT_RESUMABLE--75f2856f36.cells.py) | 0 | 0 |
| [V7F_2xRTX4080_SERVER_READY--0435f8b7fd.ipynb](notebooks/V7F_2xRTX4080_SERVER_READY--0435f8b7fd.ipynb) | [Code cells](notebooks/V7F_2xRTX4080_SERVER_READY--0435f8b7fd.cells.py) | 0 | 0 |
| [V7F_2xRTX4080_STRICT_HOLDOUT_FIXED_READY--e544391b62.ipynb](notebooks/V7F_2xRTX4080_STRICT_HOLDOUT_FIXED_READY--e544391b62.ipynb) | [Code cells](notebooks/V7F_2xRTX4080_STRICT_HOLDOUT_FIXED_READY--e544391b62.cells.py) | 0 | 0 |
| [V7F_2xRTX4080_TRUE_STAIRS_140GB_SERVER_FINAL_RESUMABLE--c71f099617.ipynb](notebooks/V7F_2xRTX4080_TRUE_STAIRS_140GB_SERVER_FINAL_RESUMABLE--c71f099617.ipynb) | [Code cells](notebooks/V7F_2xRTX4080_TRUE_STAIRS_140GB_SERVER_FINAL_RESUMABLE--c71f099617.cells.py) | 0 | 0 |
| [V7F_APPEND_GAP_SWEEP_AND_UNSEEN_TESTS--e5531bf02d.ipynb](notebooks/V7F_APPEND_GAP_SWEEP_AND_UNSEEN_TESTS--e5531bf02d.ipynb) | [Code cells](notebooks/V7F_APPEND_GAP_SWEEP_AND_UNSEEN_TESTS--e5531bf02d.cells.py) | 0 | 0 |
| [V7F_EXACT_RUN_NOTEBOOK-1--b77549b242.ipynb](notebooks/V7F_EXACT_RUN_NOTEBOOK-1--b77549b242.ipynb) | [Code cells](notebooks/V7F_EXACT_RUN_NOTEBOOK-1--b77549b242.cells.py) | 1 | 1 |
| [V7_FULL_4_BRANCH_COLAB_PY313_FIXED--92e10efd0f.ipynb](notebooks/V7_FULL_4_BRANCH_COLAB_PY313_FIXED--92e10efd0f.ipynb) | [Code cells](notebooks/V7_FULL_4_BRANCH_COLAB_PY313_FIXED--92e10efd0f.cells.py) | 0 | 0 |
| [V7_FULL_4_BRANCH_COLAB_PY313_FIXED-1--da4baec294.ipynb](notebooks/V7_FULL_4_BRANCH_COLAB_PY313_FIXED-1--da4baec294.ipynb) | [Code cells](notebooks/V7_FULL_4_BRANCH_COLAB_PY313_FIXED-1--da4baec294.cells.py) | 16 | 1 |
| [V7_FULL_4_BRANCH_END_TO_END_COLAB_READY--687f035ed6.ipynb](notebooks/V7_FULL_4_BRANCH_END_TO_END_COLAB_READY--687f035ed6.ipynb) | [Code cells](notebooks/V7_FULL_4_BRANCH_END_TO_END_COLAB_READY--687f035ed6.cells.py) | 0 | 0 |
| [V7_SHORT_FOOTHOLD_MEMORY_SERVER_RUN--b3f31589ca.ipynb](notebooks/V7_SHORT_FOOTHOLD_MEMORY_SERVER_RUN--b3f31589ca.ipynb) | [Code cells](notebooks/V7_SHORT_FOOTHOLD_MEMORY_SERVER_RUN--b3f31589ca.cells.py) | 0 | 1 |
| [V8_CONTACTMAP_STANDALONE_FRESH_COLAB--ccbaa75437.ipynb](notebooks/V8_CONTACTMAP_STANDALONE_FRESH_COLAB--ccbaa75437.ipynb) | [Code cells](notebooks/V8_CONTACTMAP_STANDALONE_FRESH_COLAB--ccbaa75437.cells.py) | 0 | 0 |
| [V8_CONTACTMAP_STANDALONE_FRESH_COLAB-1--c378f6be38.ipynb](notebooks/V8_CONTACTMAP_STANDALONE_FRESH_COLAB-1--c378f6be38.ipynb) | [Code cells](notebooks/V8_CONTACTMAP_STANDALONE_FRESH_COLAB-1--c378f6be38.cells.py) | 16 | 3 |
| [V8_CONTACTMAP_STANDALONE_FRESH_COLAB_-1--e2e3c3355c.ipynb](notebooks/V8_CONTACTMAP_STANDALONE_FRESH_COLAB_-1--e2e3c3355c.ipynb) | [Code cells](notebooks/V8_CONTACTMAP_STANDALONE_FRESH_COLAB_-1--e2e3c3355c.cells.py) | 20 | 1 |
| [V8_CONTACTMAP_STANDALONE_FRESH_COLAB_-1-3--4c7a4fc96e.ipynb](notebooks/V8_CONTACTMAP_STANDALONE_FRESH_COLAB_-1-3--4c7a4fc96e.ipynb) | [Code cells](notebooks/V8_CONTACTMAP_STANDALONE_FRESH_COLAB_-1-3--4c7a4fc96e.cells.py) | 29 | 1 |
| [V9_contact_r1_t_f--aefd0434b0.ipynb](notebooks/V9_contact_r1_t_f--aefd0434b0.ipynb) | [Code cells](notebooks/V9_contact_r1_t_f--aefd0434b0.cells.py) | 5 | 1 |
| [_v7f_rtx_executed_notebook--872c698f38.ipynb](notebooks/_v7f_rtx_executed_notebook--872c698f38.ipynb) | [Code cells](notebooks/_v7f_rtx_executed_notebook--872c698f38.cells.py) | 0 | 1 |
| [deviated_run--805534064e.ipynb](notebooks/deviated_run--805534064e.ipynb) | [Code cells](notebooks/deviated_run--805534064e.cells.py) | 6 | 1 |
| [deviated_run-1--6246e8f0aa.ipynb](notebooks/deviated_run-1--6246e8f0aa.ipynb) | [Code cells](notebooks/deviated_run-1--6246e8f0aa.cells.py) | 20 | 4 |
| [exact_architecture-2--37255bb55e.ipynb](notebooks/exact_architecture-2--37255bb55e.ipynb) | [Code cells](notebooks/exact_architecture-2--37255bb55e.cells.py) | 1 | 2 |
| [exact_architecture-3--ac372b5f53.ipynb](notebooks/exact_architecture-3--ac372b5f53.ipynb) | [Code cells](notebooks/exact_architecture-3--ac372b5f53.cells.py) | 2 | 2 |
| [exact_architecture-4--be427322b8.ipynb](notebooks/exact_architecture-4--be427322b8.ipynb) | [Code cells](notebooks/exact_architecture-4--be427322b8.cells.py) | 3 | 2 |
| [exact_architecture-6--39cfdcf5c8.ipynb](notebooks/exact_architecture-6--39cfdcf5c8.ipynb) | [Code cells](notebooks/exact_architecture-6--39cfdcf5c8.cells.py) | 4 | 3 |
| [exact_architecture_CLEAN_FIXED--393852644a.ipynb](notebooks/exact_architecture_CLEAN_FIXED--393852644a.ipynb) | [Code cells](notebooks/exact_architecture_CLEAN_FIXED--393852644a.cells.py) | 0 | 0 |
| [exact_architecture_CLEAN_FIXED_-3--ed9db6de8d.ipynb](notebooks/exact_architecture_CLEAN_FIXED_-3--ed9db6de8d.ipynb) | [Code cells](notebooks/exact_architecture_CLEAN_FIXED_-3--ed9db6de8d.cells.py) | 10 | 1 |
| [exact_architecture_with_training_and_test_metrics--9c9b3a078b.ipynb](notebooks/exact_architecture_with_training_and_test_metrics--9c9b3a078b.ipynb) | [Code cells](notebooks/exact_architecture_with_training_and_test_metrics--9c9b3a078b.cells.py) | 4 | 3 |
| [luna_fromsol_run--f2c8975527.ipynb](notebooks/luna_fromsol_run--f2c8975527.ipynb) | [Code cells](notebooks/luna_fromsol_run--f2c8975527.cells.py) | 0 | 0 |
| [prototype_v3_1--ad1608264b.ipynb](notebooks/prototype_v3_1--ad1608264b.ipynb) | [Code cells](notebooks/prototype_v3_1--ad1608264b.cells.py) | 12 | 1 |
| [prototype_v3_1_multiterrain_risk_experiment--5e1a647341.ipynb](notebooks/prototype_v3_1_multiterrain_risk_experiment--5e1a647341.ipynb) | [Code cells](notebooks/prototype_v3_1_multiterrain_risk_experiment--5e1a647341.cells.py) | 12 | 1 |
| [prototype_v3_1_multiterrain_risk_experiment-3--013d79bcb2.ipynb](notebooks/prototype_v3_1_multiterrain_risk_experiment-3--013d79bcb2.ipynb) | [Code cells](notebooks/prototype_v3_1_multiterrain_risk_experiment-3--013d79bcb2.cells.py) | 17 | 2 |
| [v4_residual_prior_direct_END_TO_END--150583500c.ipynb](notebooks/v4_residual_prior_direct_END_TO_END--150583500c.ipynb) | [Code cells](notebooks/v4_residual_prior_direct_END_TO_END--150583500c.cells.py) | 0 | 0 |
| [v4_residual_prior_direct_END_TO_END-1--7752c6b5d6.ipynb](notebooks/v4_residual_prior_direct_END_TO_END-1--7752c6b5d6.ipynb) | [Code cells](notebooks/v4_residual_prior_direct_END_TO_END-1--7752c6b5d6.cells.py) | 6 | 0 |
| [V10R_TONIGHT_RESPONSE_WORLD_MODEL1_1_2-1-.ipynb](packages/UGP_MODAL_H100_BUNDLE--2aed4827/V10R_TONIGHT_RESPONSE_WORLD_MODEL1_1_2-1-.ipynb) | [Code cells](packages/UGP_MODAL_H100_BUNDLE--2aed4827/V10R_TONIGHT_RESPONSE_WORLD_MODEL1_1_2-1-.cells.py) | 43 | 3 |
| [V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_CHECKPOINTED_FINAL.ipynb](packages/UGP_RTX_Audit_Evidence--43768345/V7F_4080_Run/V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_CHECKPOINTED_FINAL.ipynb) | [Code cells](packages/UGP_RTX_Audit_Evidence--43768345/V7F_4080_Run/V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_CHECKPOINTED_FINAL.cells.py) | 0 | 1 |
| [SOURCE_ONLY_NOTEBOOK_SNAPSHOT.ipynb](packages/UGP_V4_V5_V6_V61_MASTER_ARCHIVE_20260905_053619--c3204eec/CODE_CELLS/SOURCE_ONLY_NOTEBOOK_SNAPSHOT.ipynb) | [Code cells](packages/UGP_V4_V5_V6_V61_MASTER_ARCHIVE_20260905_053619--c3204eec/CODE_CELLS/SOURCE_ONLY_NOTEBOOK_SNAPSHOT.cells.py) | 0 | 0 |
| [C7R_SPATIAL_CAUSAL_VISUAL_REPAIR_VALIDATION_ONLY.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/C7R_SPATIAL_CAUSAL_VISUAL_REPAIR_VALIDATION_ONLY.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/C7R_SPATIAL_CAUSAL_VISUAL_REPAIR_VALIDATION_ONLY.cells.py) | 0 | 0 |
| [C7S_CAUSAL_HIGHPASS_ALIGNMENT_REPAIR_APPEND.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/C7S_CAUSAL_HIGHPASS_ALIGNMENT_REPAIR_APPEND.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/C7S_CAUSAL_HIGHPASS_ALIGNMENT_REPAIR_APPEND.cells.py) | 0 | 0 |
| [C7T_STRICT_VISION_INTERACTION_APPEND.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/C7T_STRICT_VISION_INTERACTION_APPEND.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/C7T_STRICT_VISION_INTERACTION_APPEND.cells.py) | 0 | 0 |
| [UGP_C7R_T4_RESCUE_100MIN_FINAL-1-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/UGP_C7R_T4_RESCUE_100MIN_FINAL-1-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/UGP_C7R_T4_RESCUE_100MIN_FINAL-1-.cells.py) | 4 | 0 |
| [UGP_C7R_T4_RESCUE_100MIN_FINAL-2-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/UGP_C7R_T4_RESCUE_100MIN_FINAL-2-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/UGP_C7R_T4_RESCUE_100MIN_FINAL-2-.cells.py) | 4 | 0 |
| [UGP_C7R_T4_RESCUE_100MIN_FINAL-3-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/UGP_C7R_T4_RESCUE_100MIN_FINAL-3-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/UGP_C7R_T4_RESCUE_100MIN_FINAL-3-.cells.py) | 5 | 0 |
| [UGP_C7R_T4_RESCUE_100MIN_FINAL-4-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/UGP_C7R_T4_RESCUE_100MIN_FINAL-4-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/UGP_C7R_T4_RESCUE_100MIN_FINAL-4-.cells.py) | 6 | 1 |
| [UGP_C7R_T4_RESCUE_100MIN_FINAL_-4-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/UGP_C7R_T4_RESCUE_100MIN_FINAL_-4-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/UGP_C7R_T4_RESCUE_100MIN_FINAL_-4-.cells.py) | 7 | 1 |
| [UGP_STAGE_D_E_MODULAR_APPEND.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/UGP_STAGE_D_E_MODULAR_APPEND.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/UGP_STAGE_D_E_MODULAR_APPEND.cells.py) | 0 | 0 |
| [V10R_COMPILED_CLEAN_G0B_BASELINE.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_COMPILED_CLEAN_G0B_BASELINE.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_COMPILED_CLEAN_G0B_BASELINE.cells.py) | 0 | 0 |
| [V10R_Canonical_Real_Implementation.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_Canonical_Real_Implementation.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_Canonical_Real_Implementation.cells.py) | 0 | 0 |
| [V10R_Causal_Contact_Patch_World_Model-1-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_Causal_Contact_Patch_World_Model-1-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_Causal_Contact_Patch_World_Model-1-.cells.py) | 14 | 0 |
| [V10R_Causal_Contact_Patch_World_Model-1-1-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_Causal_Contact_Patch_World_Model-1-1-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_Causal_Contact_Patch_World_Model-1-1-.cells.py) | 14 | 0 |
| [V10R_Causal_Contact_Patch_World_Model-2-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_Causal_Contact_Patch_World_Model-2-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_Causal_Contact_Patch_World_Model-2-.cells.py) | 0 | 0 |
| [V10R_Causal_Contact_Patch_World_Model.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_Causal_Contact_Patch_World_Model.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_Causal_Contact_Patch_World_Model.cells.py) | 0 | 0 |
| [V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-4-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-4-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-4-.cells.py) | 25 | 3 |
| [V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-7-1-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-7-1-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-7-1-.cells.py) | 39 | 4 |
| [V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-7-2-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-7-2-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_Causal_Contact_Patch_World_Model_-1-_-1-_-7-2-.cells.py) | 40 | 4 |
| [V10R_FINAL_END_TO_END_COLAB-2-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_FINAL_END_TO_END_COLAB-2-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_FINAL_END_TO_END_COLAB-2-.cells.py) | 51 | 0 |
| [V10R_REAL_END_TO_END_TRAINING_AND_RESULTS-1-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_REAL_END_TO_END_TRAINING_AND_RESULTS-1-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_REAL_END_TO_END_TRAINING_AND_RESULTS-1-.cells.py) | 0 | 0 |
| [V10R_REAL_END_TO_END_TRAINING_AND_RESULTS.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_REAL_END_TO_END_TRAINING_AND_RESULTS.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_REAL_END_TO_END_TRAINING_AND_RESULTS.cells.py) | 0 | 0 |
| [V10R_TONIGHT_RESPONSE_WORLD_MODEL.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_TONIGHT_RESPONSE_WORLD_MODEL.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_TONIGHT_RESPONSE_WORLD_MODEL.cells.py) | 0 | 0 |
| [V10R_TONIGHT_RESPONSE_WORLD_MODEL1_1_1.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_TONIGHT_RESPONSE_WORLD_MODEL1_1_1.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_TONIGHT_RESPONSE_WORLD_MODEL1_1_1.cells.py) | 38 | 0 |
| [V10R_TONIGHT_RESPONSE_WORLD_MODEL1_1_2-1-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_TONIGHT_RESPONSE_WORLD_MODEL1_1_2-1-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10R_TONIGHT_RESPONSE_WORLD_MODEL1_1_2-1-.cells.py) | 43 | 3 |
| [V10_GEOMETRY_COUPLED_FORCE_END_TO_END-1-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10_GEOMETRY_COUPLED_FORCE_END_TO_END-1-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10_GEOMETRY_COUPLED_FORCE_END_TO_END-1-.cells.py) | 5 | 1 |
| [V10_GEOMETRY_COUPLED_FORCE_END_TO_END-2-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10_GEOMETRY_COUPLED_FORCE_END_TO_END-2-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10_GEOMETRY_COUPLED_FORCE_END_TO_END-2-.cells.py) | 26 | 2 |
| [V10_GEOMETRY_COUPLED_FORCE_END_TO_END.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10_GEOMETRY_COUPLED_FORCE_END_TO_END.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V10_GEOMETRY_COUPLED_FORCE_END_TO_END.cells.py) | 5 | 1 |
| [V7_FULL_4_BRANCH_COLAB_PY313_FIXED-1-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V7_FULL_4_BRANCH_COLAB_PY313_FIXED-1-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V7_FULL_4_BRANCH_COLAB_PY313_FIXED-1-.cells.py) | 16 | 1 |
| [V7_FULL_4_BRANCH_COLAB_PY313_FIXED.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V7_FULL_4_BRANCH_COLAB_PY313_FIXED.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V7_FULL_4_BRANCH_COLAB_PY313_FIXED.cells.py) | 0 | 0 |
| [V7_FULL_4_BRANCH_END_TO_END_COLAB_READY.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V7_FULL_4_BRANCH_END_TO_END_COLAB_READY.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V7_FULL_4_BRANCH_END_TO_END_COLAB_READY.cells.py) | 0 | 0 |
| [V8_CONTACTMAP_STANDALONE_FRESH_COLAB-1-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V8_CONTACTMAP_STANDALONE_FRESH_COLAB-1-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V8_CONTACTMAP_STANDALONE_FRESH_COLAB-1-.cells.py) | 16 | 3 |
| [V8_CONTACTMAP_STANDALONE_FRESH_COLAB.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V8_CONTACTMAP_STANDALONE_FRESH_COLAB.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V8_CONTACTMAP_STANDALONE_FRESH_COLAB.cells.py) | 0 | 0 |
| [V8_CONTACTMAP_STANDALONE_FRESH_COLAB_-1-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V8_CONTACTMAP_STANDALONE_FRESH_COLAB_-1-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V8_CONTACTMAP_STANDALONE_FRESH_COLAB_-1-.cells.py) | 20 | 1 |
| [V8_CONTACTMAP_STANDALONE_FRESH_COLAB_-1-1-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V8_CONTACTMAP_STANDALONE_FRESH_COLAB_-1-1-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V8_CONTACTMAP_STANDALONE_FRESH_COLAB_-1-1-.cells.py) | 29 | 1 |
| [V8_CONTACTMAP_STANDALONE_FRESH_COLAB_-1-2-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V8_CONTACTMAP_STANDALONE_FRESH_COLAB_-1-2-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V8_CONTACTMAP_STANDALONE_FRESH_COLAB_-1-2-.cells.py) | 29 | 1 |
| [V8_CONTACTMAP_STANDALONE_FRESH_COLAB_-1-3-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V8_CONTACTMAP_STANDALONE_FRESH_COLAB_-1-3-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V8_CONTACTMAP_STANDALONE_FRESH_COLAB_-1-3-.cells.py) | 29 | 1 |
| [V9_contact_r1_t_f.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V9_contact_r1_t_f.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/V9_contact_r1_t_f.cells.py) | 5 | 1 |
| [deviated_run-1-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/deviated_run-1-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/deviated_run-1-.cells.py) | 20 | 4 |
| [deviated_run.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/deviated_run.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/deviated_run.cells.py) | 6 | 1 |
| [luna_fromsol_run.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/luna_fromsol_run.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/luna_fromsol_run.cells.py) | 26 | 0 |
| [prototype_v3_1-1-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/prototype_v3_1-1-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/prototype_v3_1-1-.cells.py) | 12 | 1 |
| [prototype_v3_1-2-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/prototype_v3_1-2-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/prototype_v3_1-2-.cells.py) | 12 | 1 |
| [prototype_v3_1.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/prototype_v3_1.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/prototype_v3_1.cells.py) | 12 | 1 |
| [prototype_v3_1_multiterrain_risk_experiment-1-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/prototype_v3_1_multiterrain_risk_experiment-1-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/prototype_v3_1_multiterrain_risk_experiment-1-.cells.py) | 12 | 1 |
| [prototype_v3_1_multiterrain_risk_experiment-2-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/prototype_v3_1_multiterrain_risk_experiment-2-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/prototype_v3_1_multiterrain_risk_experiment-2-.cells.py) | 13 | 2 |
| [prototype_v3_1_multiterrain_risk_experiment-3-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/prototype_v3_1_multiterrain_risk_experiment-3-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/prototype_v3_1_multiterrain_risk_experiment-3-.cells.py) | 17 | 2 |
| [prototype_v3_1_multiterrain_risk_experiment.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/prototype_v3_1_multiterrain_risk_experiment.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/prototype_v3_1_multiterrain_risk_experiment.cells.py) | 12 | 1 |
| [v4_residual_prior_direct_END_TO_END-1-.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/v4_residual_prior_direct_END_TO_END-1-.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/v4_residual_prior_direct_END_TO_END-1-.cells.py) | 6 | 0 |
| [v4_residual_prior_direct_END_TO_END.ipynb](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/v4_residual_prior_direct_END_TO_END.ipynb) | [Code cells](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/v4_residual_prior_direct_END_TO_END.cells.py) | 0 | 0 |
| [foundation.ipynb](packages/V7Feedback_28trajectory_code--5257babd/v7legs_body/foundation.ipynb) | [Code cells](packages/V7Feedback_28trajectory_code--5257babd/v7legs_body/foundation.cells.py) | 0 | 1 |
| [foundation.ipynb](packages/V7JointSpecialists_20261008_source--f7b4769f/v7legs_body/foundation.ipynb) | [Code cells](packages/V7JointSpecialists_20261008_source--f7b4769f/v7legs_body/foundation.cells.py) | 0 | 1 |
| [foundation.ipynb](packages/V7Legs_V7B_28trajectory_code--c4fcab68/foundation.ipynb) | [Code cells](packages/V7Legs_V7B_28trajectory_code--c4fcab68/foundation.cells.py) | 0 | 1 |
| [foundation.ipynb](packages/V7VisualHistory_20261010_source--448426d5/v7legs_body/foundation.ipynb) | [Code cells](packages/V7VisualHistory_20261010_source--448426d5/v7legs_body/foundation.cells.py) | 0 | 1 |
| [V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_SHORT_FOOTHOLD_MEMORY.ipynb](packages/V7_SHORT_FOOTHOLD_MEMORY_RESULTS--3c54eadc/V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_SHORT_FOOTHOLD_MEMORY.ipynb) | [Code cells](packages/V7_SHORT_FOOTHOLD_MEMORY_RESULTS--3c54eadc/V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_SHORT_FOOTHOLD_MEMORY.cells.py) | 0 | 0 |
| [V7_SHORT_FOOTHOLD_MEMORY_SERVER_RUN.ipynb](packages/V7_SHORT_FOOTHOLD_MEMORY_RESULTS--3c54eadc/V7_SHORT_FOOTHOLD_MEMORY_SERVER_RUN.ipynb) | [Code cells](packages/V7_SHORT_FOOTHOLD_MEMORY_RESULTS--3c54eadc/V7_SHORT_FOOTHOLD_MEMORY_SERVER_RUN.cells.py) | 0 | 1 |
| [foundation.ipynb](../v7-legs-body/src/v7legs_body/foundation.ipynb) | [Code cells](../v7-legs-body/src/v7legs_body/foundation.cells.py) | 0 | 1 |

## Expanded source packages

These are historical package snapshots, not independently verified new runs. Earlier model variants, V9/V10/V10R, Modal H100 launch code, terrain-model revisions and manuscript/architecture source retain their original project context. Treat their scientific claims as historical or proposed unless supported by the main evidence record.

| Source package | Files |
|---|---:|
| [Online_Visuo_Proprioceptive_Quadruped_Digital_Twin_UGP_Overleaf--ca022aa0](packages/Online_Visuo_Proprioceptive_Quadruped_Digital_Twin_UGP_Overleaf--ca022aa0/) | 11 |
| [Terrain_Foothold_Collaboration_Preflighted--e829cef9](packages/Terrain_Foothold_Collaboration_Preflighted--e829cef9/) | 7 |
| [Terrain_Foothold_Collaboration_Setup--d3c19bf5](packages/Terrain_Foothold_Collaboration_Setup--d3c19bf5/) | 6 |
| [Terrain_Foothold_RTX_Pilot_Source_Clean--297abe74](packages/Terrain_Foothold_RTX_Pilot_Source_Clean--297abe74/) | 8 |
| [UGP_Architecture_Flowchart_Overleaf--2c3203aa](packages/UGP_Architecture_Flowchart_Overleaf--2c3203aa/) | 1 |
| [UGP_MODAL_H100_BUNDLE--2aed4827](packages/UGP_MODAL_H100_BUNDLE--2aed4827/) | 5 |
| [UGP_RTX_Audit_Evidence--43768345](packages/UGP_RTX_Audit_Evidence--43768345/) | 184 |
| [UGP_V4_V5_V6_V61_MASTER_ARCHIVE_20260905_053619--c3204eec](packages/UGP_V4_V5_V6_V61_MASTER_ARCHIVE_20260905_053619--c3204eec/) | 2 |
| [V10R_FINAL_DIAGNOSTIC_RESULTS_AND_MODEL--11c04d0d](packages/V10R_FINAL_DIAGNOSTIC_RESULTS_AND_MODEL--11c04d0d/) | 3 |
| [V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3](packages/V10R_STAGE_B_B1_B10_APPEND_CELLS--dcda12f3/) | 51 |
| [V7F_4080_Run_scripts--4eeaab3b](packages/V7F_4080_Run_scripts--4eeaab3b/) | 4 |
| [V7F_4080_Run_scripts_v2--6c5e6bcd](packages/V7F_4080_Run_scripts_v2--6c5e6bcd/) | 4 |
| [V7F_4080_Run_scripts_v3--d3637c61](packages/V7F_4080_Run_scripts_v3--d3637c61/) | 5 |
| [V7F_FORCE_SPECIALIST_RESULTS--aba02085](packages/V7F_FORCE_SPECIALIST_RESULTS--aba02085/) | 1 |
| [V7Feedback_28trajectory_code--5257babd](packages/V7Feedback_28trajectory_code--5257babd/) | 19 |
| [V7JointSpecialists_20261008_source--f7b4769f](packages/V7JointSpecialists_20261008_source--f7b4769f/) | 42 |
| [V7Legs_V7B_28trajectory_code--c4fcab68](packages/V7Legs_V7B_28trajectory_code--c4fcab68/) | 11 |
| [V7VisualHistory_20261010_source--448426d5](packages/V7VisualHistory_20261010_source--448426d5/) | 14 |
| [V7_All_Architectures_Presentation_Images_20261009--09a99c1c](packages/V7_All_Architectures_Presentation_Images_20261009--09a99c1c/) | 5 |
| [V7_SHORT_FOOTHOLD_MEMORY_RESULTS--3c54eadc](packages/V7_SHORT_FOOTHOLD_MEMORY_RESULTS--3c54eadc/) | 3 |
| [V7_terrain_grounded_runnable--ed02a468](packages/V7_terrain_grounded_runnable--ed02a468/) | 11 |
| [merged_feelwm_v7_ssh_project--f8b46429](packages/merged_feelwm_v7_ssh_project--f8b46429/) | 14 |
| [tartanground_10ep_update--efb9261e](packages/tartanground_10ep_update--efb9261e/) | 9 |
| [tartanground_k1_k5_leg_compare--08249ca5](packages/tartanground_k1_k5_leg_compare--08249ca5/) | 47 |
| [tartanground_p2000_k1_results_step200--0a066bc5](packages/tartanground_p2000_k1_results_step200--0a066bc5/) | 30 |

CS781 source, private data, checkpoints, credentials, installed Python environments and upstream dependency checkouts are excluded. Obtain external foundations from their owners under their own terms. The inaccessible RTX server may contain additional versions not recoverable from this local snapshot.
