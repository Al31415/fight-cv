# Fight-CV Event Ontology — Data Dictionary (v1.0.0, FROZEN)

Machine copy: [ontology_v1.yaml](ontology_v1.yaml). This document is the authority
for **mutually exclusive** definitions. Freeze this before any annotation; bump
`version` and re-freeze for any change. Every event/effect/state row records the
`ontology_version` it was created under.

## Design rules (non-negotiable)

1. **One attempt event per attempt**, carrying a result — never separate
   "attempt" and "landed" atomic events (that double-counts).
2. **Effects are measured, not judged.** Knockdown/stumble/posture-break/cut are
   observable effects, stored separately, optionally linked to the causing attempt
   via `source_event_id`.
3. **Damage, momentum, fatigue are latent interpretations** modeled downstream
   with uncertainty — never raw annotation labels. Fatigue is only ever stored as
   phase-conditioned proxies.
4. **Observability is first-class.** If an outcome is not observable, annotate
   `visibility` accordingly and set `result=unknown`; evaluation excludes
   genuinely unobservable outcomes or scores the model's ability to abstain.

## Attempt events

An **attempt** is a single committed offensive action by one fighter (`actor_uid`)
toward the other (`target_uid`). Fields: `action_family`, `action_subtype`,
`result`, `round`, `clock`, `source_pts`, `shot_id`, `phase`, `exchange_id`,
plus observability + provenance.

| family | subtypes | result vocabulary | result definitions |
|---|---|---|---|
| **strike** | jab, cross, hook, uppercut, overhand, elbow, low_kick, body_kick, head_kick, front_kick, knee, spinning_strike, ground_strike, other_strike | landed / blocked / missed / unknown | **landed** = makes contact with the target's body/head not behind a blocking limb; **blocked** = contact absorbed by guard/limb intentionally interposed; **missed** = no contact; **unknown** = attempt visible but outcome not observable |
| **takedown** | double_leg, single_leg, trip, throw, slam, drag, other_takedown | completed / stuffed / failed / abandoned / unknown | **completed** = actor brings target to the mat and establishes top/controlling position; **stuffed** = defended before actor achieves control (target stays up or reverses); **failed** = attempt initiated, target grounded but no control gained; **abandoned** = actor disengages before resolution; **unknown** = not observable |
| **submission** | rear_naked, guillotine, armbar, triangle, kimura, americana, kneebar, heel_hook, dArce, anaconda, other_submission | secured / defended / escaped / abandoned / unknown | **secured** = hold applied with finishing pressure (tap/technical finish or clear locked position); **defended** = target prevents the hold from being applied; **escaped** = hold applied then target frees; **abandoned** = actor releases; **unknown** = not observable |

A strike **attempt** exists the moment the limb is committed; the `result`
resolves its outcome. Do not emit a second event for the landing.

## Effects (separate table)

Measured consequences, each with `source_event_id?` (nullable link to the causing
attempt), `magnitude?` (numeric where measurable), and observability.

| type | definition |
|---|---|
| **knockdown** | target is put down (not via takedown) by a strike; touches mat with hand/knee/body or drops |
| **stumble** | clear loss of balance without going down |
| **posture_break** | in clinch/ground, target's posture is broken/forced down |
| **cut** | visible laceration/bleeding appears |
| **head_displacement** | measurable head snap/displacement from a strike (magnitude = px or normalized) |
| **visible_wobble** | unsteady legs/compromised motor control after impact |

## Phase (interval state — mutually exclusive at any instant)

`standing`, `clinch`, `ground`, `transition`, `break`, `unknown`. When
`ground`, a `ground_position` ∈ {guard, half_guard, side_control, mount, back,
turtle, sprawl, scramble, other_ground}. `control` ∈ {none, top_control,
back_control, clinch_control, unknown}. Phase intervals tile the fight timeline
without overlap.

## Shot / broadcast state (Stage 00)

`content` ∈ {live, replay, slow_motion, picture_in_picture, graphic, crowd,
corner, walkout, replay_review, other}. Only `live` (and optionally `slow_motion`
when adjudicating) feeds authoritative event counts.

## Observability, review status, provenance

- `visibility` ∈ {clear, partial, occluded, off_camera}; `annotation_confidence`
  ∈ [0,1]; `annotator_id`; `ontology_version`.
- `review_status` ∈ {raw, candidate, auto_accepted, human_adjudicated, rejected}.
  A VLM may **suggest/supply evidence**; a **human** produces adjudicated
  development/test labels; a calibrated policy may **auto_accept** production
  events (retaining `auto_accepted`) — VLM-assisted never equals human truth.
- `abstention_reason` ∈ {low_confidence, occluded, off_camera,
  ambiguous_identity, out_of_ontology, insufficient_evidence}.
