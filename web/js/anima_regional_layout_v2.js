import { app } from "../../../scripts/app.js";

const EDITOR_WIDTH = 460;
const EDITOR_HEIGHT = 246;
const EDITOR_BASE_NODE_HEIGHT = 610;
const EDITOR_MAX_CANVAS_HEIGHT = 420;
const GRID_STEP = 1 / 32;
const MAX_CHARACTERS = 8;
const PALETTE = ["#E7584B", "#35A7D8", "#44B86B", "#E9A43A", "#9C65D1", "#D75C9A", "#35B9B1", "#B8BF42"];
const REGION_TYPES = ["body_region", "ownership_hint"];

function uuid(prefix) {
  if (globalThis.crypto?.randomUUID) return `${prefix}-${globalThis.crypto.randomUUID()}`;
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function clamp(value, low, high) {
  return Math.min(high, Math.max(low, Number.isFinite(+value) ? +value : low));
}

function widget(node, name) {
  return node.widgets?.find((item) => item.name === name);
}

function widgetValue(node, name, fallback = "") {
  return widget(node, name)?.value ?? fallback;
}

function imageAspect(node) {
  const imageWidth = Math.max(1, Number(widgetValue(node, "width", 1024)));
  const imageHeight = Math.max(1, Number(widgetValue(node, "height", 1024)));
  return imageWidth / imageHeight;
}

function editorHeightForNode(node) {
  const nodeWidth = Math.max(EDITOR_WIDTH, Number(node.size?.[0]) || EDITOR_WIDTH);
  const availableWidth = Math.max(120, nodeWidth - 16);
  const baseCanvasHeight = EDITOR_HEIGHT - 72;
  const targetCanvasHeight = Math.min(
    EDITOR_MAX_CANVAS_HEIGHT,
    Math.max(baseCanvasHeight, availableWidth / imageAspect(node)),
  );
  return targetCanvasHeight + 72;
}

function resizeEditorNode(node) {
  const editorHeight = editorHeightForNode(node);
  node._animaRegionalEditorHeight = editorHeight;
  const targetNodeHeight = Math.max(
    EDITOR_BASE_NODE_HEIGHT,
    EDITOR_BASE_NODE_HEIGHT + (editorHeight - EDITOR_HEIGHT),
  );
  if (Math.abs((node.size?.[1] || 0) - targetNodeHeight) > 1) {
    node.size[1] = targetNodeHeight;
    node.graph?.setDirtyCanvas?.(true, true);
  }
}

function hideWidget(item) {
  if (!item || item._animaHidden) return;
  item._animaHidden = true;
  item.hidden = true;
  item.computeSize = () => [0, 0];
  item.draw = () => {};
}

function parseLayout(value) {
  try {
    const layout = JSON.parse(String(value || ""));
    if (layout && typeof layout === "object" && !Array.isArray(layout)) return layout;
  } catch (_) {
    // A hand-edited incomplete JSON is restored to a valid editor payload.
  }
  return { version: 2, regions: [] };
}

function stableColor(id) {
  let value = 0;
  for (let index = 0; index < id.length; index += 1) value = ((value * 31) + id.charCodeAt(index)) >>> 0;
  return PALETTE[value % PALETTE.length];
}

function backendCharacterUuid(nodeId) {
  if (nodeId == null || nodeId === -1 || String(nodeId).trim() === "") return null;
  const safe = String(nodeId).trim().replace(/[^A-Za-z0-9._:-]+/g, "-").replace(/^[-._:]+|[-._:]+$/g, "");
  return safe ? `character-node-${safe}`.slice(0, 128) : null;
}

function characterUuid(source) {
  let idWidget = widget(source, "character_uuid");
  if (!idWidget) {
    // Some ComfyUI builds omit optional STRING widgets until a frontend adds one.
    // Add the declared input by name so the backend receives the same value too.
    idWidget = source.addWidget?.("text", "character_uuid", uuid("character"), null, { serialize: true });
    if (!idWidget) {
      source.properties ??= {};
      source.properties._animaRegionalCharacterUuid ??= uuid("character");
      const stable = backendCharacterUuid(source.id);
      if (stable && stable !== source.properties._animaRegionalCharacterUuid) {
        source._animaRegionalPreviousUuids ??= new Set();
        source._animaRegionalPreviousUuids.add(source.properties._animaRegionalCharacterUuid);
        source.properties._animaRegionalCharacterUuid = stable;
      }
      return source.properties._animaRegionalCharacterUuid;
    }
  }
  const previous = String(idWidget.value || "").trim();
  const resolved = backendCharacterUuid(source.id) || previous || uuid("character");
  if (previous && previous !== resolved) {
    source._animaRegionalPreviousUuids ??= new Set();
    source._animaRegionalPreviousUuids.add(previous);
  }
  idWidget.value = resolved;
  hideWidget(idWidget);
  return resolved;
}

function connectedCharacters(node) {
  const graph = node.graph;
  if (!graph) return [];
  const links = graph.links || {};
  const sources = [];
  for (let index = 1; index <= MAX_CHARACTERS; index += 1) {
    const input = node.inputs?.find((item) => item.name === `character_${index}`);
    if (!input || input.link == null) continue;
    const link = links instanceof Map ? links.get(input.link) : links[input.link];
    const source = graph.getNodeById?.(link?.origin_id) || graph._nodes_by_id?.[link?.origin_id];
    if (!source || source.comfyClass !== "AnimaRegionalCharacterPromptV2") continue;
    const id = characterUuid(source);
    sources.push({
      slot: index,
      uuid: id,
      label: String(widgetValue(source, "label", `Character ${index}`)).trim() || `Character ${index}`,
      prompt: String(widgetValue(source, "prompt", "")),
      strength: clamp(widgetValue(source, "strength", 1), 0, 4),
      color: String(widgetValue(source, "color", "")).trim() || stableColor(id),
      source,
      aliases: [...(source._animaRegionalPreviousUuids || [])],
    });
  }
  return sources;
}

function syncCharacterInputs(node) {
  const state = stateFor(node);
  if (!state.connectionsSettled || node._animaRegionalConfiguring || node._animaRegionalSyncingInputs) return;
  node._animaRegionalSyncingInputs = true;
  try {
    const slots = (node.inputs || [])
      .map((input, index) => ({ input, index }))
      .filter(({ input }) => /^character_[1-8]$/.test(input.name));
    const connected = slots.filter(({ input }) => input.link != null);
    const empty = slots.filter(({ input }) => input.link == null);
    for (const { index } of empty.sort((a, b) => b.index - a.index)) node.removeInput?.(index);

    if (connected.length < MAX_CHARACTERS) {
      const usedNames = new Set(connected.map(({ input }) => input.name));
      let nextName = null;
      for (let index = 1; index <= MAX_CHARACTERS; index += 1) {
        if (!usedNames.has(`character_${index}`)) {
          nextName = `character_${index}`;
          break;
        }
      }
      if (nextName) node.addInput?.(nextName, "ANIMA_REGIONAL_CHARACTER_V2");
    }
    node.graph?.setDirtyCanvas?.(true, true);
  } finally {
    node._animaRegionalSyncingInputs = false;
  }
}

function scheduleCharacterInputSync(node) {
  clearTimeout(node._animaRegionalSocketTimer);
  node._animaRegionalSocketTimer = setTimeout(() => syncCharacterInputs(node), 0);
}

function regionRecord(character, type = "body_region") {
  return {
    uuid: uuid("region"),
    character_uuid: character.uuid,
    type,
    geometry: "box",
    x: 0.25,
    y: 0.2,
    width: 0.5,
    height: 0.6,
    feather: 0,
    enabled: true,
  };
}

function stateFor(node) {
  if (node._animaRegionalLayoutState) return node._animaRegionalLayoutState;
  const layout = parseLayout(widgetValue(node, "layout_json", ""));
  const state = {
    regions: Array.isArray(layout.regions) ? layout.regions.map((item) => ({ ...item })) : [],
    characters: Array.isArray(layout.characters) ? layout.characters.map((item) => ({ ...item })) : [],
    orphaned: new Map(
      (Array.isArray(layout.orphaned_regions) ? layout.orphaned_regions : [])
        .filter((item) => item && item.uuid)
        .map((item) => [item.uuid, { ...item }]),
    ),
    selected: null,
    zoom: 1,
    grid: true,
    hover: null,
    drag: null,
    lastHighlighted: null,
    connectionsSettled: false,
    changeDepth: 0,
  };
  node._animaRegionalLayoutState = state;
  return state;
}

function currentRegion(node) {
  const state = stateFor(node);
  return state.regions.find((item) => item.uuid === state.selected) || null;
}

function beginGraphChange(node) {
  const state = stateFor(node);
  if (state.changeDepth === 0) node.graph?.beforeChange?.();
  state.changeDepth += 1;
}

function endGraphChange(node) {
  const state = stateFor(node);
  if (state.changeDepth <= 0) return;
  state.changeDepth -= 1;
  if (state.changeDepth === 0) node.graph?.afterChange?.();
}

function graphChange(node, callback) {
  beginGraphChange(node);
  try {
    return callback();
  } finally {
    endGraphChange(node);
  }
}

function clearHighlight(node) {
  const state = node._animaRegionalLayoutState;
  const source = state?.lastHighlighted;
  if (source?._animaRegionalLayoutHighlights) {
    source._animaRegionalLayoutHighlights.delete(node);
    if (!source._animaRegionalLayoutHighlights.size) delete source._animaRegionalLayoutHighlights;
  }
  if (state) state.lastHighlighted = null;
}

function syncHighlight(node) {
  const state = stateFor(node);
  const region = currentRegion(node);
  const source = region && connectedCharacters(node).find((item) => item.uuid === region.character_uuid)?.source;
  if (state.lastHighlighted !== source) clearHighlight(node);
  if (source) {
    source._animaRegionalLayoutHighlights ??= new Set();
    source._animaRegionalLayoutHighlights.add(node);
    state.lastHighlighted = source;
  }
}

function setJson(node) {
  const state = stateFor(node);
  const connected = connectedCharacters(node);
  const connectedIds = new Set(connected.map((item) => item.uuid));
  for (const character of connected) {
    for (const alias of character.aliases) {
      if (!alias || connectedIds.has(alias)) continue;
      for (const region of state.regions) if (region.character_uuid === alias) region.character_uuid = character.uuid;
      for (const region of state.orphaned.values()) if (region.character_uuid === alias) region.character_uuid = character.uuid;
    }
  }
  const chars = connected.length ? connected : (state.connectionsSettled ? [] : state.characters);
  if (connected.length || state.connectionsSettled) state.characters = connected.map(({ uuid: id, label, prompt, strength, color }) => ({
    version: 2, kind: "character", uuid: id, label, prompt, strength, color,
  }));
  const active = new Set(chars.map((item) => item.uuid));
  const authoritative = connected.length > 0 || state.connectionsSettled;
  const retained = authoritative ? [] : state.regions.slice();
  if (authoritative) for (const region of state.regions) {
    if (active.has(region.character_uuid)) retained.push(region);
    else state.orphaned.set(region.uuid, { ...region });
  }
  const present = new Set(retained.map((item) => item.uuid));
  for (const [id, region] of authoritative ? state.orphaned : []) {
    if (active.has(region.character_uuid) && !present.has(id)) {
      retained.push({ ...region });
      state.orphaned.delete(id);
    }
  }
  state.regions = retained;
  if (!state.regions.some((item) => item.uuid === state.selected)) state.selected = state.regions[0]?.uuid || null;

  const payload = {
    version: 2,
    width: Math.round(clamp(widgetValue(node, "width", 1024), 64, 8192)),
    height: Math.round(clamp(widgetValue(node, "height", 1024), 64, 8192)),
    overlap_mode: String(widgetValue(node, "overlap_mode", "exclusive")),
    characters: state.characters.map((item) => ({ ...item })),
    regions: state.regions.map((region) => ({
      uuid: region.uuid,
      character_uuid: region.character_uuid,
      type: REGION_TYPES.includes(region.type) ? region.type : "body_region",
      geometry: "box",
      x: clamp(region.x, 0, 1),
      y: clamp(region.y, 0, 1),
      width: clamp(region.width, 0.001, 1),
      height: clamp(region.height, 0.001, 1),
      feather: clamp(region.feather, 0, 1),
      enabled: Boolean(region.enabled),
    })),
    orphaned_regions: [...state.orphaned.values()].map((region) => ({
      uuid: region.uuid,
      character_uuid: region.character_uuid,
      type: REGION_TYPES.includes(region.type) ? region.type : "body_region",
      geometry: "box",
      x: clamp(region.x, 0, 1),
      y: clamp(region.y, 0, 1),
      width: clamp(region.width, 0.001, 1),
      height: clamp(region.height, 0.001, 1),
      feather: clamp(region.feather, 0, 1),
      enabled: Boolean(region.enabled),
    })),
  };
  for (const region of [...payload.regions, ...payload.orphaned_regions]) {
    region.width = Math.min(region.width, 1 - region.x);
    region.height = Math.min(region.height, 1 - region.y);
  }
  const jsonWidget = widget(node, "layout_json");
  if (jsonWidget) jsonWidget.value = JSON.stringify(payload);
  syncHighlight(node);
  refreshControls(node);
  node.graph?.setDirtyCanvas?.(true, true);
}

function selectRegion(node, id) {
  const state = stateFor(node);
  state.selected = id;
  syncHighlight(node);
  refreshControls(node);
  node.graph?.setDirtyCanvas?.(true, true);
}

function addControlWidgets(node) {
  if (node._animaRegionalControlsAdded || !node.addWidget) return;
  node._animaRegionalControlsAdded = true;
  const change = (name, value) => {
    const region = currentRegion(node);
    if (!region) return;
    graphChange(node, () => {
      if (name === "character") {
        const choice = stateFor(node).choices?.get(value);
        if (choice) region.character_uuid = choice.uuid;
      } else if (name === "enabled") region.enabled = Boolean(value);
      else if (name === "type") region.type = value;
      else region[name] = clamp(value, 0, 1);
      setJson(node);
    });
  };
  const editorWidget = (kind, name, label, value, callback, options = {}) => {
    const item = node.addWidget(kind, `anima_editor_${name}`, value, callback, options);
    item.label = label;
    item.serialize = false;
    item.options ??= {};
    item.options.serialize = false;
    return item;
  };
  const character = editorWidget("combo", "character", "body character", "", (value) => change("character", value), { values: [] });
  character.tooltip = "Body region: choose the character whose identity owns this box. Ownership Hint: choose the local owner; it does not create another prompt.";
  const type = editorWidget("combo", "type", "region type", "body_region", (value) => change("type", value), { values: REGION_TYPES });
  const number = (name, label) => editorWidget("number", name, label, 0, (value) => change(name, value), { min: 0, max: 1, step: 0.005, precision: 3 });
  const controls = {
    character,
    type,
    x: number("x", "x"), y: number("y", "y"), width: number("width", "width"), height: number("height", "height"),
    feather: number("feather", "feather"),
    enabled: editorWidget("toggle", "enabled", "enabled", true, (value) => change("enabled", value)),
    zoom: editorWidget("number", "zoom", "editor zoom", 1, (value) => { stateFor(node).zoom = clamp(value, 0.5, 2); node.graph?.setDirtyCanvas?.(true, true); }, { min: 0.5, max: 2, step: 0.1 }),
    grid: editorWidget("toggle", "grid", "snap to grid", true, (value) => { stateFor(node).grid = Boolean(value); node.graph?.setDirtyCanvas?.(true, true); }),
  };
  node._animaRegionalControls = controls;
}

function refreshControls(node) {
  const controls = node._animaRegionalControls;
  if (!controls) return;
  const state = stateFor(node);
  const region = currentRegion(node);
  const chars = connectedCharacters(node);
  state.choices = new Map(chars.map((item) => [`${item.label} (${item.slot})`, item]));
  controls.character.options.values = [...state.choices.keys()];
  if (!region) return;
  const selectedChar = chars.find((item) => item.uuid === region.character_uuid);
  const isHint = region.type === "ownership_hint";
  controls.character.label = isHint ? "owner character" : "body character";
  controls.character.tooltip = isHint
    ? "Ownership Hint owner: this local box borrows the selected character's existing prompt. It is active only in exclusive mode."
    : "Body region character: this box defines the selected character's broad spatial area.";
  controls.character.value = selectedChar ? `${selectedChar.label} (${selectedChar.slot})` : "";
  controls.type.value = region.type;
  for (const key of ["x", "y", "width", "height", "feather", "enabled"]) controls[key].value = region[key];
  controls.zoom.value = state.zoom;
  controls.grid.value = state.grid;
}

function canvasGeometry(width, y, node) {
  const availableWidth = Math.max(120, width - 16);
  const editorHeight = node._animaRegionalEditorHeight || editorHeightForNode(node);
  const availableHeight = Math.max(120, editorHeight - 72);
  const imageWidth = Math.max(1, Number(widgetValue(node, "width", 1024)));
  const imageHeight = Math.max(1, Number(widgetValue(node, "height", 1024)));
  const aspect = imageWidth / imageHeight;
  let canvasWidth = availableWidth;
  let canvasHeight = canvasWidth / aspect;
  if (canvasHeight > availableHeight) {
    canvasHeight = availableHeight;
    canvasWidth = canvasHeight * aspect;
  }
  return {
    x: 8 + ((availableWidth - canvasWidth) / 2),
    y: y + 31 + ((availableHeight - canvasHeight) / 2),
    width: canvasWidth,
    height: canvasHeight,
    imageWidth,
    imageHeight,
    aspect,
  };
}

function regionRect(region, canvas, zoom) {
  const scaledWidth = canvas.width * zoom;
  const scaledHeight = canvas.height * zoom;
  const offsetX = canvas.x - ((scaledWidth - canvas.width) / 2);
  const offsetY = canvas.y - ((scaledHeight - canvas.height) / 2);
  return { x: offsetX + (region.x * scaledWidth), y: offsetY + (region.y * scaledHeight), width: region.width * scaledWidth, height: region.height * scaledHeight };
}

function hitMode(rect, x, y) {
  const pad = 7;
  if (x < rect.x - pad || x > rect.x + rect.width + pad || y < rect.y - pad || y > rect.y + rect.height + pad) return null;
  const left = Math.abs(x - rect.x) <= pad;
  const right = Math.abs(x - rect.x - rect.width) <= pad;
  const top = Math.abs(y - rect.y) <= pad;
  const bottom = Math.abs(y - rect.y - rect.height) <= pad;
  if (left && top) return "nw";
  if (right && top) return "ne";
  if (left && bottom) return "sw";
  if (right && bottom) return "se";
  if (left) return "w";
  if (right) return "e";
  if (top) return "n";
  if (bottom) return "s";
  return "move";
}

function toolbarHit(x, y) {
  if (y < 3 || y > 27) return null;
  const index = Math.floor((x - 8) / 28);
  return ["body", "hint", "copy", "delete"][index] || null;
}

function pointerEventKind(event) {
  const type = String(event?.type || "").toLowerCase();
  if (type === "mousedown" || type === "pointerdown") return "down";
  if (type === "mousemove" || type === "pointermove") return "move";
  if (type === "mouseup" || type === "pointerup") return "up";
  if (type === "mouseleave" || type === "pointerleave" || type === "pointercancel") return "cancel";
  return type;
}

function snapToGrid(value, enabled) {
  return enabled ? Math.round(value / GRID_STEP) * GRID_STEP : value;
}

function action(node, kind) {
  graphChange(node, () => {
    const state = stateFor(node);
    const chars = connectedCharacters(node);
    const selected = currentRegion(node);
    if ((kind === "body" || kind === "hint") && chars.length) {
      const character = chars.find((item) => item.uuid === selected?.character_uuid) || chars[0];
      const region = regionRecord(character, kind === "hint" ? "ownership_hint" : "body_region");
      if (selected) { region.x = clamp(selected.x + 0.04, 0, 0.9); region.y = clamp(selected.y + 0.04, 0, 0.9); }
      state.regions.push(region);
      selectRegion(node, region.uuid);
    } else if (kind === "copy" && selected) {
      const copy = { ...selected, uuid: uuid("region"), x: clamp(selected.x + 0.035, 0, 0.99 - selected.width), y: clamp(selected.y + 0.035, 0, 0.99 - selected.height) };
      state.regions.push(copy);
      selectRegion(node, copy.uuid);
    } else if (kind === "delete" && selected) {
      state.regions = state.regions.filter((item) => item.uuid !== selected.uuid);
      state.selected = state.regions[0]?.uuid || null;
    }
    setJson(node);
  });
}

function drawLayoutEditor(ctx, node, width, y) {
  const state = stateFor(node);
  const canvas = canvasGeometry(width, y, node);
  const chars = connectedCharacters(node);
  const colors = new Map(chars.map((item) => [item.uuid, item.color || stableColor(item.uuid)]));
  ctx.save();
  ctx.font = "13px sans-serif";
  const labels = [["+", "Add body region"], ["!", "Add Ownership Hint"], ["=", "Copy selected region"], ["x", "Delete selected region"]];
  labels.forEach(([icon, tip], index) => {
    const x = 8 + (index * 28);
    const hovered = state.hover === ["body", "hint", "copy", "delete"][index];
    ctx.fillStyle = hovered ? "#58606e" : "#39414d";
    ctx.fillRect(x, y + 4, 22, 20);
    ctx.fillStyle = "#e6edf3";
    ctx.fillText(icon, x + 8, y + 19);
    if (hovered) {
      ctx.fillStyle = "#15191f";
      ctx.fillRect(x, y + 27, Math.max(122, ctx.measureText(tip).width + 12), 20);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(tip, x + 6, y + 42);
    }
  });
  ctx.fillStyle = "#20252d";
  ctx.fillRect(canvas.x, canvas.y, canvas.width, canvas.height);
  ctx.save();
  ctx.beginPath(); ctx.rect(canvas.x, canvas.y, canvas.width, canvas.height); ctx.clip();
  if (state.grid) {
    ctx.strokeStyle = "rgba(185, 197, 211, 0.16)";
    ctx.lineWidth = 1;
    for (let step = 0; step <= 8; step += 1) {
      const px = canvas.x + ((step / 8) * canvas.width);
      const py = canvas.y + ((step / 8) * canvas.height);
      ctx.beginPath(); ctx.moveTo(px, canvas.y); ctx.lineTo(px, canvas.y + canvas.height); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(canvas.x, py); ctx.lineTo(canvas.x + canvas.width, py); ctx.stroke();
    }
  }
  for (const region of state.regions) {
    const rect = regionRect(region, canvas, state.zoom);
    const selected = region.uuid === state.selected;
    const color = colors.get(region.character_uuid) || stableColor(region.character_uuid);
    ctx.fillStyle = `${color}${region.enabled ? "35" : "14"}`;
    ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
    ctx.strokeStyle = selected ? "#ffffff" : color;
    ctx.lineWidth = selected ? 2 : 1;
    if (region.type === "ownership_hint") ctx.setLineDash([5, 3]); else ctx.setLineDash([]);
    ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
    ctx.setLineDash([]);
    if (selected) {
      ctx.fillStyle = "#ffffff";
      for (const point of [[rect.x, rect.y], [rect.x + rect.width, rect.y], [rect.x, rect.y + rect.height], [rect.x + rect.width, rect.y + rect.height]]) ctx.fillRect(point[0] - 3, point[1] - 3, 6, 6);
    }
  }
  ctx.restore();
  ctx.strokeStyle = "#6a7380";
  ctx.strokeRect(canvas.x, canvas.y, canvas.width, canvas.height);
  ctx.fillStyle = "#b8c0ca";
  const selected = currentRegion(node);
  const selectedCharacter = selected && chars.find((item) => item.uuid === selected.character_uuid);
  const overlapMode = String(widgetValue(node, "overlap_mode", "exclusive"));
  const selectedLabel = selectedCharacter ? selectedCharacter.label : "unassigned";
  const regionText = selected
    ? `${selected.type.replace(/_/g, " ")} | ${selectedLabel} | ${selected.uuid.slice(-8)}`
    : "Connect a Character Prompt V2, then add a region.";
  const editorHeight = node._animaRegionalEditorHeight || editorHeightForNode(node);
  ctx.fillText(regionText, 8, y + editorHeight - 20);
  ctx.fillStyle = selected?.type === "ownership_hint" && overlapMode !== "exclusive" ? "#e9a43a" : "#8f9baa";
  const modeText = selected?.type === "ownership_hint"
    ? (overlapMode === "exclusive" ? "Ownership Hint active in exclusive mode" : "Ownership Hint inactive in normalized mode")
    : `Frame ${canvas.imageWidth}x${canvas.imageHeight} | aspect ${canvas.aspect.toFixed(3)} | overlap ${overlapMode}`;
  ctx.fillText(modeText, 8, y + editorHeight - 5);
  ctx.restore();
}

function mouseLayoutEditor(node, event, pos, width, y) {
  const state = stateFor(node);
  const x = pos?.[0]; const mouseY = pos?.[1];
  if (!Number.isFinite(x) || !Number.isFinite(mouseY)) return false;
  const canvas = canvasGeometry(width, y, node);
  const eventKind = pointerEventKind(event);
  if (eventKind === "move") {
    state.hover = toolbarHit(x, mouseY - y);
    if (!state.drag) { node.graph?.setDirtyCanvas?.(true, false); return Boolean(state.hover); }
    const region = currentRegion(node);
    if (!region) return false;
    const scaleX = canvas.width * state.zoom; const scaleY = canvas.height * state.zoom;
    const dx = (x - state.drag.x) / scaleX; const dy = (mouseY - state.drag.y) / scaleY;
    const initial = state.drag.initial;
    let left = initial.x; let top = initial.y; let right = initial.x + initial.width; let bottom = initial.y + initial.height;
    const mode = state.drag.mode;
    if (mode === "move") { left += dx; right += dx; top += dy; bottom += dy; }
    else {
      if (mode.includes("w")) left += dx;
      if (mode.includes("e")) right += dx;
      if (mode.includes("n")) top += dy;
      if (mode.includes("s")) bottom += dy;
    }
    const minSize = state.grid ? GRID_STEP : 0.01;
    if (mode === "move") {
      left = clamp(snapToGrid(left, state.grid), 0, 1 - initial.width);
      top = clamp(snapToGrid(top, state.grid), 0, 1 - initial.height);
      right = left + initial.width;
      bottom = top + initial.height;
    } else {
      if (mode.includes("w")) left = snapToGrid(left, state.grid);
      if (mode.includes("e")) right = snapToGrid(right, state.grid);
      if (mode.includes("n")) top = snapToGrid(top, state.grid);
      if (mode.includes("s")) bottom = snapToGrid(bottom, state.grid);
      left = clamp(left, 0, right - minSize);
      top = clamp(top, 0, bottom - minSize);
      right = clamp(right, left + minSize, 1);
      bottom = clamp(bottom, top + minSize, 1);
    }
    Object.assign(region, { x: left, y: top, width: right - left, height: bottom - top });
    setJson(node);
    return true;
  }
  if (eventKind === "down") {
    const control = toolbarHit(x, mouseY - y);
    if (control) { action(node, control); return true; }
    for (const region of [...state.regions].reverse()) {
      const mode = hitMode(regionRect(region, canvas, state.zoom), x, mouseY);
      if (mode) {
        selectRegion(node, region.uuid);
        beginGraphChange(node);
        state.drag = { x, y: mouseY, mode, initial: { ...region }, changeOpen: true };
        return true;
      }
    }
  }
  if (eventKind === "up" || eventKind === "cancel") {
    if (state.drag?.changeOpen) endGraphChange(node);
    state.drag = null;
    node.graph?.setDirtyCanvas?.(true, true);
    return true;
  }
  return false;
}

function addCanvasWidget(node) {
  if (node._animaRegionalCanvasAdded) return;
  node._animaRegionalCanvasAdded = true;
  const custom = {
    name: "anima_regional_layout_canvas",
    type: "anima_regional_layout_canvas",
    serialize: false,
    options: { serialize: false },
    computeSize: (width) => [
      Math.max(EDITOR_WIDTH - 20, width || 0),
      node._animaRegionalEditorHeight || EDITOR_HEIGHT,
    ],
    draw: (ctx, owner, width, y) => {
      custom._drawWidth = width;
      custom._drawY = y;
      drawLayoutEditor(ctx, owner, width, y);
    },
    mouse: (event, pos, owner) => mouseLayoutEditor(owner, event, pos, custom._drawWidth || owner.size?.[0] || EDITOR_WIDTH, custom._drawY || 0),
  };
  if (node.addCustomWidget) node.addCustomWidget(custom); else node.widgets.push(custom);
}

function settleLayoutConnections(node) {
  clearTimeout(node._animaRegionalSettleTimer);
  node._animaRegionalSettleTimer = setTimeout(() => {
    if (app.configuringGraph || node._animaRegionalConfiguring) {
      settleLayoutConnections(node);
      return;
    }
    const state = stateFor(node);
    state.connectionsSettled = true;
    setJson(node);
    syncCharacterInputs(node);
  }, 16);
}

function configureLayoutNode(node) {
  if (node._animaRegionalLayoutConfigured) return;
  node._animaRegionalLayoutConfigured = true;
  const jsonWidget = widget(node, "layout_json");
  hideWidget(jsonWidget);
  addControlWidgets(node);
  addCanvasWidget(node);
  node.size[0] = Math.max(node.size[0] || 0, EDITOR_WIDTH);
  node.size[1] = Math.max(node.size[1] || 0, EDITOR_BASE_NODE_HEIGHT);
  resizeEditorNode(node);
  const originalConnections = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = originalConnections?.apply(this, args);
    if (!this._animaRegionalSyncingInputs) {
      setJson(this);
      settleLayoutConnections(this);
      scheduleCharacterInputSync(this);
    }
    return result;
  };
  const originalConfigureMethod = node.configure;
  node.configure = function (...args) {
    clearHighlight(this);
    this._animaRegionalConfiguring = true;
    try {
      return originalConfigureMethod?.apply(this, args);
    } finally {
      this._animaRegionalLayoutState = null;
      this._animaRegionalConfiguring = false;
      settleLayoutConnections(this);
    }
  };
  const originalConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalConfigure?.apply(this, args);
    clearHighlight(this);
    this._animaRegionalLayoutState = null;
    settleLayoutConnections(this);
    return result;
  };
  const originalRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    if (stateFor(this).drag?.changeOpen) endGraphChange(this);
    clearTimeout(this._animaRegionalSettleTimer);
    clearTimeout(this._animaRegionalSocketTimer);
    clearHighlight(this);
    return originalRemoved?.apply(this, args);
  };
  ["width", "height", "overlap_mode"].forEach((name) => {
    const item = widget(node, name);
    if (!item || item._animaLayoutHooked) return;
    item._animaLayoutHooked = true;
    const callback = item.callback;
    item.callback = function (value, ...args) {
      const result = callback?.call(this, value, ...args);
      setJson(node);
      resizeEditorNode(node);
      return result;
    };
  });
  setJson(node);
  settleLayoutConnections(node);
}

app.registerExtension({
  name: "anima.regional.layout-v2",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "AnimaRegionalCharacterPromptV2") {
      if (nodeType.prototype._animaRegionalCharacterHooksInstalled) return;
      nodeType.prototype._animaRegionalCharacterHooksInstalled = true;
      const created = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function (...args) {
        const result = created?.apply(this, args);
        characterUuid(this);
        return result;
      };
      const draw = nodeType.prototype.onDrawForeground;
      nodeType.prototype.onDrawForeground = function (ctx, ...args) {
        draw?.call(this, ctx, ...args);
        if (!this._animaRegionalLayoutHighlights?.size) return;
        ctx.save(); ctx.strokeStyle = "#f2d35c"; ctx.lineWidth = 3;
        ctx.strokeRect(2, 2, this.size[0] - 4, this.size[1] - 4); ctx.restore();
      };
    }
    if (nodeData.name !== "AnimaRegionalLayoutV2") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = created?.apply(this, args);
      configureLayoutNode(this);
      return result;
    };
  },
});

function setPromptPackWidgetPresentation(node) {
  const presentation = {
    global_prompt: [
      "shared scene context",
      "Shared scene description. Connect Anima Regional - Shared Scene Prompt to reuse it in Artist Pack and this Prompt Pack.",
    ],
    negative_prompt: ["negative prompt", "Optional text negative prompt. It is ignored when external negative conditioning is connected."],
    base_positive: ["external base positive (Mixer output)", "Connect the Post-Adapter Mixer's final positive conditioning here."],
    base_negative: ["external negative conditioning", "Optional external negative conditioning. When connected, the text negative prompt is ignored."],
  };
  for (const [name, [label, tooltip]] of Object.entries(presentation)) {
    const item = widget(node, name);
    if (!item) continue;
    item.label = label;
    item.tooltip = tooltip;
  }
}

app.registerExtension({
  name: "anima.regional.prompt-pack-v2-presentation",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "AnimaRegionalPromptPackV2" && nodeData.name !== "AnimaRegionalSharedPromptV2") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = created?.apply(this, args);
      if (nodeData.name === "AnimaRegionalPromptPackV2") setPromptPackWidgetPresentation(this);
      else {
        const item = widget(this, "scene_prompt");
        if (item) {
          item.label = "shared scene prompt";
          item.tooltip = "Enter the shared scene description once, then connect this STRING output to Artist Pack and Regional Prompt Pack.";
        }
      }
      return result;
    };
  },
});
