# GenesisBench — landing page

Single-file static site (`index.html` + `assets/hero.png`). Layout, type, and palette follow
posttrain.com (source: `benchflow-ai/posttrain-arena` — Satoshi, warm paper `#fbf9f2`,
forest-green primary, terracotta accent).

```bash
cd website && python3 -m http.server 8000   # http://localhost:8000/
```

## Hero artwork

A hand-drawn storybook-watercolor "march of progress" of robot embodiment (generated with Azure
`gpt-image-2`), full-bleed behind the hero like posttrain's terrain. Each figure is drawn from a
real, publicly-demonstrated machine:

| # | Figure | Real robot |
|---|--------|-----------|
| 1 | early boxy service robot with bellows arm | early industrial/service robots (Unimate lineage, 1961→) |
| 2 | two dexterous robotic hands (one holding a bulb) | Shadow Dexterous Hand-class anthropomorphic hands |
| 3 | dual-arm headless torso, arms raised | ABB YuMi (IRB 14000, 2015) |
| 4 | torso + sensor head on wheeled base | Willow Garage PR2 (2010) |
| 5 | caged bipedal humanoid, exposed actuators | Boston Dynamics Atlas (2013) |
| 6 | sleek humanoid, black faceplate, silver chest | Tesla Optimus (2023) |

The page content (idea → Provision/Train/Evaluate/Transfer → frameworks → research) follows the
repo README: GenesisBench 1.0 — language intelligence → physical intelligence; 2.0 — world
intelligence.
