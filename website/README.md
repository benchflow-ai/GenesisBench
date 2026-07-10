# GenesisBench — landing page

Single-file static site (`index.html` + `assets/hero.png`). Structure and type follow
posttrain.com (source: `benchflow-ai/posttrain-arena` — Satoshi, 1280px container, terrain-style
hero), re-paletted to **monochrome**: warm pencil-paper background, black/grey ink, no color.

```bash
cd website && python3 -m http.server 8000   # http://localhost:8000/
```

## Hero artwork

A pencil "march of progress" of robot embodiment — six figures evolving low→high, far→near,
left→right: a simple machine → an industrial arm → walking legs → dual arms → a torso humanoid →
a full humanoid. Composed from the project's own six pencil illustrations with Azure `gpt-image-2`
(depth-recession layout), full-bleed behind the hero like posttrain's terrain.

Page content follows the repo README: the idea (Provision → Train → Evaluate → Transfer),
frameworks (Genesis, Isaac Sim, RoboCasa, MuJoCo), research background. GenesisBench 1.0 —
language intelligence → physical intelligence; 2.0 — world intelligence.
