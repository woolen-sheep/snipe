let api = null;
const photo = document.getElementById("photo");
const overlays = document.getElementById("overlays");
const canvas = document.getElementById("canvas");
const viewer = document.getElementById("viewer");
const fileNameEl = document.getElementById("fileName");
const filePathEl = document.getElementById("filePath");
const statusEl = document.getElementById("statusText");
const prevBtn = document.getElementById("btnPrev");
const nextBtn = document.getElementById("btnNext");
const resetBtn = document.getElementById("btnReset");

let state = null;
let scale = 1;
let origin = { x: 0, y: 0 };
let isPanning = false;
let isAutoFit = true;
let panStart = { x: 0, y: 0 };
let resizeFrame = null;
const MIN_ZOOM = 0.2;
const MAX_ZOOM = 6;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function applyTransform() {
  canvas.style.transform = `translate(${origin.x}px, ${origin.y}px) scale(${scale})`;
}

function centerCanvas(width, height) {
  const rect = viewer.getBoundingClientRect();
  const fitScale = Math.min(rect.width / width, rect.height / height);
  // Default to 1:1 unless the image is larger than the viewport; then scale down to fit.
  scale = clamp(Math.min(1, fitScale), MIN_ZOOM, MAX_ZOOM);
  origin = {
    x: (rect.width - width * scale) / 2,
    y: (rect.height - height * scale) / 2,
  };
  applyTransform();
}

function setStatus(message) {
  statusEl.textContent = message;
}

function renderRegions(payload) {
  overlays.innerHTML = "";
  overlays.style.width = `${payload.imageWidth}px`;
  overlays.style.height = `${payload.imageHeight}px`;

  payload.regions.forEach((region) => {
    const box = document.createElement("div");
    box.className = "box";
    box.style.left = `${region.x}px`;
    box.style.top = `${region.y}px`;
    box.style.width = `${region.width}px`;
    box.style.height = `${region.height}px`;

    const label = document.createElement("div");
    label.className = "label";
    label.textContent = region.label;

    const edge = Math.max(24, region.width);
    const fontSize = Math.max(14 / scale, Math.max(payload.imageWidth * 0.05, edge * 0.12) / region.label.length);
    label.style.fontSize = `${fontSize}px`;
    label.style.padding = `${Math.max(3, fontSize * 0.28)}px ${Math.max(8, fontSize * 0.7)}px`;
    label.style.borderRadius = `${Math.max(6, fontSize * 0.4)}px`;
    console.log(`Region "${region.label}" font size: ${fontSize}px edge: ${edge}px scale: ${scale}`);

    box.appendChild(label);
    overlays.appendChild(box);
  });
}

function render(payload) {
  state = payload;
  isPanning = false;
  isAutoFit = true;
  scale = 1;
  origin = { x: 0, y: 0 };
  fileNameEl.textContent = `${payload.fileName} (${payload.index + 1} / ${payload.total})`;
  filePathEl.textContent = payload.path;
  setStatus("Use arrow keys or buttons to navigate. Scroll to zoom, drag to pan.");

  overlays.innerHTML = "";
  photo.style.visibility = "hidden";
  applyTransform();

  photo.onload = () => {
    centerCanvas(payload.imageWidth, payload.imageHeight);
    renderRegions(payload);
    setStatus(`${payload.regions.length} region(s) loaded. Scroll to zoom, drag to pan.`);
    photo.style.visibility = "visible";
  };

  photo.onerror = () => {
    setStatus("Failed to load image. Check logs for details.");
  };

  photo.src = payload.imageUri;
  photo.style.width = `${payload.imageWidth}px`;
  photo.style.height = `${payload.imageHeight}px`;
  canvas.style.width = `${payload.imageWidth}px`;
  canvas.style.height = `${payload.imageHeight}px`;
}

async function loadInitial() {
  const payload = await api.current();
  console.log("Loaded initial payload", payload);
  render(payload);
}

async function loadNext() {
  const payload = await api.next();
  console.log("Loaded next payload", payload);
  render(payload);
}

async function loadPrev() {
  const payload = await api.previous();
  console.log("Loaded previous payload", payload);
  render(payload);
}

function resetView() {
  if (!state) return;
  isAutoFit = true;
  centerCanvas(state.imageWidth, state.imageHeight);
  renderRegions(state);
}

function handleWheel(event) {
  event.preventDefault();
  if (!state) return;
  isAutoFit = false;

  const rect = viewer.getBoundingClientRect();
  const offsetX = event.clientX - rect.left;
  const offsetY = event.clientY - rect.top;
  const zoomFactor = event.deltaY < 0 ? 1.1 : 0.9;
  const newScale = clamp(scale * zoomFactor, MIN_ZOOM, MAX_ZOOM);
  const ratio = newScale / scale;

  origin.x = offsetX - (offsetX - origin.x) * ratio;
  origin.y = offsetY - (offsetY - origin.y) * ratio;
  scale = newScale;
  clampPan();
  applyTransform();
}

function startPan(event) {
  if (event.button !== 0) return;
  isAutoFit = false;
  isPanning = true;
  viewer.classList.add("dragging");
  panStart = {
    x: event.clientX - origin.x,
    y: event.clientY - origin.y,
  };
}

function movePan(event) {
  if (!isPanning) return;
  origin.x = event.clientX - panStart.x;
  origin.y = event.clientY - panStart.y;
  clampPan();
  applyTransform();
}

function endPan() {
  isPanning = false;
  viewer.classList.remove("dragging");
}

function clampPan() {
  if (!state) return;
  const rect = viewer.getBoundingClientRect();
  const imgW = state.imageWidth * scale;
  const imgH = state.imageHeight * scale;

  // If image fits entirely, center it and stop here
  if (imgW <= rect.width) {
    origin.x = (rect.width - imgW) / 2;
  } else {
    const minX = rect.width - imgW;
    const maxX = 0;
    origin.x = clamp(origin.x, minX, maxX);
  }

  if (imgH <= rect.height) {
    origin.y = (rect.height - imgH) / 2;
  } else {
    const minY = rect.height - imgH;
    const maxY = 0;
    origin.y = clamp(origin.y, minY, maxY);
  }
}

function handleResize() {
  if (!state) return;

  if (resizeFrame !== null) {
    window.cancelAnimationFrame(resizeFrame);
  }

  resizeFrame = window.requestAnimationFrame(() => {
    resizeFrame = null;

    if (isAutoFit) {
      centerCanvas(state.imageWidth, state.imageHeight);
      renderRegions(state);
      return;
    }

    clampPan();
    applyTransform();
    renderRegions(state);
  });
}

window.addEventListener("keydown", (event) => {
  if (event.key === "ArrowRight") {
    loadNext();
  } else if (event.key === "ArrowLeft") {
    loadPrev();
  }
});

viewer.addEventListener("wheel", handleWheel, { passive: false });
viewer.addEventListener("mousedown", startPan);
window.addEventListener("mousemove", movePan);
window.addEventListener("mouseup", endPan);
window.addEventListener("resize", handleResize);

prevBtn.addEventListener("click", loadPrev);
nextBtn.addEventListener("click", loadNext);
resetBtn.addEventListener("click", resetView);

async function waitForApi(timeoutMs = 5000) {
  if (window.pywebview && window.pywebview.api) {
    api = window.pywebview.api;
    return api;
  }

  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("pywebview api not ready")), timeoutMs);
    const ready = () => {
      if (window.pywebview && window.pywebview.api) {
        api = window.pywebview.api;
        clearTimeout(timer);
        resolve(api);
      }
    };
    window.addEventListener("pywebviewready", ready, { once: true });
    // Poll as a fallback in case the event is missed
    const poll = setInterval(() => {
      if (window.pywebview && window.pywebview.api) {
        api = window.pywebview.api;
        clearTimeout(timer);
        clearInterval(poll);
        resolve(api);
      }
    }, 100);
  });
}

async function bootstrap() {
  try {
    await waitForApi();
    await loadInitial();
  } catch (error) {
    console.error(error);
    setStatus("Failed to load images. pywebview API not ready.");
  }
}

bootstrap();
