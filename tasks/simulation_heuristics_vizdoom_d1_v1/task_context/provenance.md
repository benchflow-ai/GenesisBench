# Provenance and licensing

## Behavioral source

The experiment definition and target metrics come from:

```text
repository: https://github.com/Trinkle23897/learning-beyond-gradients
revision: 3555c2956c257d49a5015b782cbe485b14fd659e
article section: VizDoom D1 Basic CV Policy
declared runtime: EnvPool 1.1.1
```

The public article describes brightness thresholding, morphology, connected
components, visual alignment, and delayed medikit collection using rendered
pixels plus public `HEALTH`.

## Reimplementation boundary

The pinned source repository did not contain a declared license. GenesisBench
therefore does not copy or redistribute its implementation. This task's API,
runtime, starter policy, evaluator, verifier, and compact reference policy are
new code written for this repository from the public behavioral description
and independently reproduced output.

EnvPool, OpenCV, NumPy, VizDoom runtime assets, and their transitive components
remain governed by their respective upstream licenses.

The source README explicitly states that the experiments were written against
EnvPool `1.1.1`. Both the source policy and the GenesisBench oracle reproduce
the published ten-seed vector on that version, so no compatibility exception
is used.
