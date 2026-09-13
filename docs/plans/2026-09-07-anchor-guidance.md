# Anchor guidance implementation plan

> Execute locally in this session using the approved design. The user requested implementation through verification; no further design approval or GitHub publication is needed.

**Goal:** Provide persistent recovery guidance and an explicit, editable automatic-anchor recovery path.

**Architecture:** A bounded on-demand recommendation service works in original image coordinates, retaining separate mask regions. Draft recommendations do not alter stored anchors. An explicit anchor registration mode estimates only a similarity transform and uses existing versioned result/history storage.

**Tech Stack:** Python, NumPy/OpenCV, existing LoFTR, FastAPI, Vue 3/TypeScript, pytest and Playwright.

## Tasks

1. Add tests in `tests/test_anchor_recovery.py` for uneven region density, real paired coordinates, missing/ambiguous support, degeneracy, ratio preservation and contradictory points. Implement `anchor_recovery.py`: independently match the selected regions, deduplicate and spatially distribute candidates, then fit explicit anchor pairs without falling back to the failing global consensus.
2. Extend `webapp/server.py` with a nonmutating recommendation endpoint, bounded cache, revision-bound draft token and anchor registration input checks. Preserve user anchors, previous results and undo. Verify API requests against isolated sessions, including stale recommendation and failed retry cases.
3. Extend `Workspace.vue` with local recommendation drafts, point editing/deletion and explicit rerun; guard asynchronous work by image pair and input revisions. Add the persistent guidance panel under comparison controls with four actions, short state messages and accessible descriptions. Connect registration to `App.vue` existing job lifecycle. Add explicit previous-result restoration using version checks.
4. Add Playwright coverage with deterministic inference only at the model boundary: suggestions appear without changing anchors/results, edits and rerun work, manual/mask/adjust actions navigate correctly, old requests cannot overwrite another photo, narrow layout and focus mode remain usable.
5. Run pytest, typecheck/unit/build and browser suites. Review screenshots and consequential diff. Start the new local version alongside the existing session if needed; verify its live identity/API and provide the new URL. Existing in-memory session must remain accessible. Notify through the authorized Telegram hook at completion.

## Scope and validation limits

User clarification: anatomical names in the motivating simulation must not become implementation rules. Treat each user-selected object as a generic reference region, distribute suggestions within each region, and prevent a dense region from consuming the entire recommendation budget. Do not infer tooth class or privilege an anterior/posterior position.

This implements the approved recovery workflow. It does not replace the full automatic registration pipeline or claim clinical validation. Recommendation confidence is a matcher/filter signal, not a probability of anatomical correctness. Anchor fitting errors are not independent validation. Missing or incompatible references remain visible; neither affine distortion nor fake projected anchor pairs are allowed.

Full implementation outcomes are tracked in `.unlazy/anchor-guidance/GATES.md` before edits. The live session is not reset or killed. Existing unrelated modifications remain intact.

## Implemented layout refinement

Visual review found that a simple in-flow footer could fall below the visible work area when anchor tools opened. The final layout keeps the guidance panel in a dedicated footer and scrolls the image/tool section above it. Rerun and cancel sit in the guidance header. Suggested-pair view scrolls to the images, while the footer remains visible. The four recovery buttons remain available in the default state and in focus mode.

Reference regions are generic: no tooth classification or anatomical location priority is used. Real cached LoFTR inference was checked on synthetic image pairs with a known translation, in addition to unit/API and browser contracts. This does not establish accuracy on clinical photos. The normal global automatic pipeline is unchanged; the new explicit recovery path uses the reviewed anchors and preserves uniform scaling.
