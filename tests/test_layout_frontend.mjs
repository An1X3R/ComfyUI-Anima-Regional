import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath, pathToFileURL } from "node:url";

const testDirectory = path.dirname(fileURLToPath(import.meta.url));
const sourcePath = path.resolve(testDirectory, "../web/js/anima_regional_layout_v2.js");
const source = fs.readFileSync(sourcePath, "utf8");
const context = vm.createContext({
  clearTimeout,
  console,
  crypto: { randomUUID: () => "test-uuid" },
  setTimeout,
});
const module = new vm.SourceTextModule(source, {
  context,
  identifier: pathToFileURL(sourcePath).href,
});
await module.link(async (specifier) => {
  if (!specifier.endsWith("/scripts/app.js")) {
    throw new Error(`Unexpected frontend import: ${specifier}`);
  }
  return new vm.SyntheticModule(["app"], function initialize() {
    this.setExport("app", { registerExtension() {} });
  }, { context });
});
await module.evaluate();

const {
  globalMixShares,
  mirroredRegionCopy,
  mirrorRegionHorizontally,
} = module.namespace;

const hint = {
  uuid: "hint-a",
  character_uuid: "kaltsit",
  type: "ownership_hint",
  x: 0.125,
  y: 0.3125,
  width: 0.25,
  height: 0.375,
  feather: 0.05,
  priority: 3,
  enabled: true,
  hint_blend: "soft",
  strength: 0.7,
};

const mirrored = mirrorRegionHorizontally(hint);
assert.equal(mirrored.x, 0.625);
assert.equal(mirrored.uuid, hint.uuid);
assert.equal(mirrored.character_uuid, hint.character_uuid);
assert.equal(mirrored.feather, hint.feather);
assert.equal(mirrored.priority, hint.priority);
assert.equal(mirrored.hint_blend, hint.hint_blend);
assert.equal(mirrored.strength, hint.strength);
assert.equal(mirrorRegionHorizontally(mirrored).x, hint.x);

const reciprocal = mirroredRegionCopy(hint, "hint-b", "virtuosa");
assert.equal(reciprocal.uuid, "hint-b");
assert.equal(reciprocal.character_uuid, "virtuosa");
assert.equal(reciprocal.x, 0.625);
assert.equal(reciprocal.feather, 0.05);
assert.equal(reciprocal.priority, 3);
assert.equal(reciprocal.hint_blend, "soft");
assert.equal(reciprocal.strength, 0.7);

assert.deepEqual(
  { ...globalMixShares(0.25, 1) },
  { baseShare: 0.2, characterShare: 0.8, basePercent: 20, characterPercent: 80 },
);
assert.deepEqual(
  { ...globalMixShares(1, 1) },
  { baseShare: 0.5, characterShare: 0.5, basePercent: 50, characterPercent: 50 },
);
assert.equal(globalMixShares(0.25, 2).basePercent, 11);

console.log("layout frontend tests passed");
