/**
 * Holds the single live NeuralController so training state persists as the user
 * navigates between the builder, the neural detail modal, and the results view.
 */

import { NeuralController } from "./neural/network";
import { getSituation } from "./data/situations";
import { store } from "./state";
import type { DatasetKind } from "./neural/dataset";

let controller: NeuralController | null = null;

function currentDataset(): DatasetKind {
  const sit = getSituation(store.get().situationId);
  return sit?.dataset ?? "circle";
}

/** Get (or lazily create) the controller, keeping it in sync with the situation. */
export function getController(): NeuralController {
  const dataset = currentDataset();
  if (!controller) {
    controller = new NeuralController(store.get().neural, dataset);
  } else if (controller.dataset !== dataset) {
    controller.config = store.get().neural;
    controller.setDataset(dataset);
  }
  return controller;
}

/** Re-sync config and rebuild the network (after a structural/config change). */
export function rebuildController(): void {
  const c = getController();
  c.config = store.get().neural;
  c.rebuild();
}

