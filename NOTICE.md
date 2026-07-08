# Third-party notices

This project vendors source code from the following projects.

## TensorFlow Playground

- Files: `src/neural/nn.ts` (and any other file marked "VENDORED from TensorFlow Playground")
- Source: https://github.com/tensorflow/playground
- Copyright 2016 Google Inc.
- License: Apache License, Version 2.0

The neural-network engine that powers the "expand the neural block" view is taken
from TensorFlow Playground so that its behavior matches the original tool. The full
text of the Apache 2.0 license is available at:

    http://www.apache.org/licenses/LICENSE-2.0

A copy is included in `LICENSE-APACHE-2.0.txt`.

## TensorFlow Playground (embedded build)

- Files: `public/tfpg/` (`index.html`, `bundle.js`, `bundle.css`, `lib.js`)
- Source: https://github.com/tensorflow/playground (compiled distribution from
  https://playground.tensorflow.org)
- Copyright 2016 Google Inc.
- License: Apache License, Version 2.0

The "expand the neural block" modal embeds the real, compiled TensorFlow Playground
in an `<iframe>` so the experience is identical to the original. `index.html` is the
upstream page with only the surrounding article/header/footer hidden and Google
Analytics removed; the JavaScript and CSS bundles are unmodified upstream builds.

