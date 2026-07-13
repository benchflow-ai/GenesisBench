# Evaluation and aggregation

For game `g`, raw episode return `R`, random anchor `r_g`, and human anchor
`h_g`:

```text
HNS(g, R) = (R - r_g) / (h_g - r_g)
```

`atari57_games.csv` contains the 57 random/human anchors inferred by the source
artifact and the known-best score supplied to each article search.

The primary aggregate is:

1. evaluate repeat-specific policy `0`, `1`, and `2` on their corresponding
   evaluation seeds;
2. compute mean HNS over those three policies for each game/mode;
3. select the higher mode mean for each game;
4. take the median over 57 games.

This is the article's **best input mean** convention. The verifier also reports:

- mean of the 57 selected mode means;
- median and mean after taking the best single evaluation run per game;
- per-game and per-mode returns/HNS;
- invalid-policy rate and counted evaluation steps;
- the separate 342-search interaction-ledger summary.

The public evaluator uses six representative games and three public seeds,
covering 36 repeat-specific slots. It is for debugging only. The checked-in
hidden reproduction config uses all 57 games, both modes, and three seeds: 342
final evaluation trajectories.

The normalized GenesisBench score uses natural numeric HNS anchors:

```text
0.0 HNS = 0
0.8283015254994576 HNS = 100
```

The second value is the article's reported Codex best-input-mean median HNS.
The seeded-random oracle artifact is not this target. Normalization does not
replace HNS; raw HNS metrics remain in the details JSON.
