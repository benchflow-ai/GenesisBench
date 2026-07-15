# GenesisBench — landing page

Single-file static site (`index.html` + `assets/hero.png`). Structure and type follow
posttrain.com (source: `benchflow-ai/posttrain-arena` — Satoshi, 1280px container, terrain-style
hero), re-paletted to **monochrome**: neutral near-white paper, black/grey ink, no color.

```bash
cd website && python3 -m http.server 8000   # http://localhost:8000/
```

## Hero artwork

A pencil "march of progress" of robot embodiment — six figures evolving low→high, far→near,
left→right: a simple machine → an industrial arm → walking legs → dual arms → a torso humanoid →
a full humanoid. Composed from the project's own six pencil illustrations with Azure `gpt-image-2`
(depth-recession layout), full-bleed behind the hero like posttrain's terrain.

Page content follows the repo README: an interactive six-stage pipeline (heuristic policies →
VLA training, simulation → real world → more hardware → 2.0) whose first stage embeds an
interactive improvement loop and navigator across all nine article-derived tasks. The selected
task updates its environment artwork, starter description, experiment snippets, scoring-suite
summary, and task links. The page also includes an environments diagram (simulation frameworks:
Genesis, Isaac Sim, RoboCasa, MuJoCo · real-world operation), native HTML article-suite
leaderboards, and the research background. GenesisBench 1.0 — language intelligence → physical
intelligence; 2.0 — world intelligence.

The website copies `leaderboard/article_suite.json` to
`website/assets/article_suite.json` and renders both plots as responsive native
HTML/CSS. No raster leaderboard image is embedded on the page.

Offline README/report images are generated separately by:

```bash
uv run python scripts/plot_article_suite_leaderboards.py
```

The HTML task panels show five-trial mean ± sample standard deviation for
native raw environment scores. The final HTML plot shows RLiable-style pooled
IQM over all 45 trial-task scores, plus the sample standard deviation of the
five per-trial IQMs, and uses a fixed positive display index (`IQM + 100`) while
retaining raw IQM as the official ranking metric.

Inference cards also display the exact provider route, OpenCode harness,
reasoning setting, and Daytona sandbox used by each published model.

## Deploy — genesisbench.benchflow.ai (Vercel, same as clawsbench)

The repo ships `vercel.json` (serve `website/` as-is, no build; deploys are skipped unless
`website/` changed). One-time setup:

The `ignoreCommand` compares `website/` and `vercel.json`. Repository-only
maintenance outside those paths is intentionally skipped and does not change
the production site.

1. **Vercel** (benchflow team) → *Add New Project* → import `benchflow-ai/GenesisBench` →
   settings are read from `vercel.json` → Deploy. From then on every push to `main` that touches
   `website/` auto-deploys to production, and every PR gets a preview URL.
2. **Vercel → Project → Settings → Domains** → add `genesisbench.benchflow.ai`. Vercel shows a
   CNAME target (e.g. `xxxx.vercel-dns-0xx.com`).
3. **GoDaddy DNS for benchflow.ai** → add record: `CNAME · genesisbench · <target from step 2>`
   (same pattern as the existing `clawsbench` record).

No changes to the `www.benchflow.ai` repo are needed — subdomains are independent Vercel projects.
