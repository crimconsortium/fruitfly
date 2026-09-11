# trace/v1

A trace is the complete record of one fly, one seed paper, one run. The engine writes
it; the renderer reads it. Nothing else passes between the two layers.

This contract is versioned so a future browser engine can replace the Python one
without touching the renderer.

```jsonc
{
  "schema": "trace/v1",
  "engine": {
    "rng_seed": 20260911,          // fly's own RNG; same seed => same trace
    "dt_ms": 1.0,
    "window_ms": 50,               // simulated time per decision
    "tau_v_ms": 20.0,
    "tau_syn_ms": 5.0,
    "v_rest": -52.0,
    "v_threshold": -45.0,
    "v_reset": -52.0,
    "gain": 0.02,
    "n_input_neurons": 800,
    "n_channels": 8,               // candidate slots, and readout groups
    "neurons": 19873,
    "edges": 1204511,
    "learning": false              // v1 is purely reactive. No plasticity.
  },
  "policy": { "open_statuses": ["diamond", "gold", "green", "hybrid"] },
  "seed": {
    "id": "W2141234567",
    "title": "...",
    "journal": "...",
    "year": 2019,
    "oa_status": "closed",
    "passable": false
  },
  "nodes": [
    {
      "id": "W2141234567",
      "title": "...",
      "journal": "...",
      "year": 2019,
      "oa_status": "gold",         // diamond|gold|green|hybrid|bronze|closed|unknown
      "passable": true,
      "terrain": "corridor"        // corridor|gate|trapdoor|wall
    }
  ],
  "steps": [
    {
      "t": 0,                      // decision index
      "at": "W2141234567",         // where the fly is standing
      "to": "W1998887766",         // where it went; null if stuck
      "candidates": [
        { "id": "W1998887766", "channel": 0, "passable": true,
          "oa_status": "green", "drive_hz": 41.2, "readout_spikes": 37 }
      ],
      "bumps": ["W2222333344"],    // walls the fly touched this step
      "total_spikes": 812,
      "backtracked": false
    }
  ],
  "result": {
    "visited": 14,
    "reachable": 14,               // size of the OA-passable closure actually entered
    "walls_hit": 63,
    "n_steps": 21,
    "stuck_at": "W1717171717",
    "stuck_reason": "no_passable_neighbours",  // or seed_paywalled | hop_ceiling
    "truncated": false
  }
}
```

## Terrain mapping

| oa_status | terrain | passable |
|---|---|---|
| diamond | corridor (pristine) | yes |
| gold | corridor | yes |
| green | corridor | yes |
| hybrid | gate | yes |
| bronze | trapdoor | **no** |
| closed | wall | no |
| unknown | wall | no |

Bronze is drawn as a visibly cracked trapdoor rather than a solid wall: it is free to
read today at the publisher's discretion, with no open license guaranteeing tomorrow.
We do not let the fly walk on it.
