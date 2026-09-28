// UI logic for CropLens. Talks to the backend only through
// window.CropLensAPI (see api.js) - never calls fetch() directly.
//
// Ported from the original single-file prototype's inline <script>, with
// two changes: (1) images are kept as File/Blob objects for upload
// instead of base64 strings, matching the real API's multipart/form-data
// /predict endpoint, and (2) renderResult() hides sections the real API
// has no data for (numeric N-P-K, growth timeline, medicinal profile)
// instead of silently showing zeros/empty content for them.

// ---------------------------------------------------------------------
// Decorative particle background (unchanged from the original)
// ---------------------------------------------------------------------
(function () {
  const canvas = document.getElementById("particleCanvas");
  const ctx = canvas.getContext("2d");
  let W, H, particles = [];
  const SYMBOLS = ["✦", "·", "◦", "❋", "∗", "⁕", "✿", "❁"];
  const COLORS = [
    "rgba(82,183,136,",
    "rgba(64,145,108,",
    "rgba(233,196,106,",
    "rgba(216,243,220,",
  ];
  function resize() {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  function createParticle() {
    const color = COLORS[Math.floor(Math.random() * COLORS.length)];
    return {
      x: Math.random() * W,
      y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.3,
      vy: -Math.random() * 0.5 - 0.1,
      size: Math.random() * 10 + 5,
      alpha: Math.random() * 0.3 + 0.05,
      alphaV: (Math.random() - 0.5) * 0.003,
      symbol: SYMBOLS[Math.floor(Math.random() * SYMBOLS.length)],
      color,
      rotation: Math.random() * Math.PI * 2,
      rotV: (Math.random() - 0.5) * 0.008,
    };
  }
  function init() {
    resize();
    const count = Math.min(55, Math.floor((W * H) / 22000));
    for (let i = 0; i < count; i++) particles.push(createParticle());
  }
  function tick() {
    ctx.clearRect(0, 0, W, H);
    for (let p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      p.alpha += p.alphaV;
      p.rotation += p.rotV;
      if (p.alpha <= 0.02 || p.alpha >= 0.38) p.alphaV *= -1;
      if (p.y < -20) {
        p.y = H + 10;
        p.x = Math.random() * W;
      }
      if (p.x < -20 || p.x > W + 20) p.x = Math.random() * W;
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rotation);
      ctx.globalAlpha = p.alpha;
      ctx.fillStyle = p.color + p.alpha + ")";
      ctx.font = p.size + "px serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(p.symbol, 0, 0);
      ctx.restore();
    }
    requestAnimationFrame(tick);
  }
  window.addEventListener("resize", resize);
  init();
  tick();
})();

// ---------------------------------------------------------------------
// State - File/Blob objects (not base64), so they can go straight into
// FormData for the real /predict endpoint.
// ---------------------------------------------------------------------
let uploadedImageFile = null; // File, from <input type="file">
let capturedImageBlob = null; // Blob, from the camera canvas
let cameraStream = null;

// ---------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------
function switchTab(tab) {
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
  document.querySelector('[data-tab="' + tab + '"]').classList.add("active");
  document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
  document.getElementById("panel-" + tab).classList.add("active");
  hideResults();
}

function hideResults() {
  ["loadingCard", "errorCard", "resultCard"].forEach((id) =>
    document.getElementById(id).classList.remove("show")
  );
  document.getElementById("resetBtn").style.display = "none";
}

// ---------------------------------------------------------------------
// Upload tab
// ---------------------------------------------------------------------
function handleFile(e) {
  const file = e.target.files[0];
  if (!file) return;

  uploadedImageFile = file; // kept as a File for upload

  const reader = new FileReader();
  reader.onload = (ev) => {
    const preview = document.getElementById("previewImg");
    preview.src = ev.target.result; // data URL, for preview only
    preview.classList.add("show");
    document.getElementById("uploadZone").style.display = "none";
    document.getElementById("analyzeUploadBtn").disabled = false;
  };
  reader.readAsDataURL(file);
}

const uploadZone = document.getElementById("uploadZone");
uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("drag-over");
});
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleFile({ target: { files: [file] } });
});

// ---------------------------------------------------------------------
// Camera tab
// ---------------------------------------------------------------------
async function startCamera() {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });
    const video = document.getElementById("videoEl");
    video.srcObject = cameraStream;
    video.style.display = "block";
    document.getElementById("camStatus").style.display = "none";
    document.getElementById("camOverlay").style.display = "flex";
    document.getElementById("snapBtn").disabled = false;
    document.getElementById("startCamBtn").disabled = true;
  } catch (err) {
    document.getElementById("camStatus").textContent =
      "Camera access denied. Please allow camera permissions.";
  }
}

function takeSnapshot() {
  const video = document.getElementById("videoEl");
  const canvas = document.getElementById("snapCanvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);

  // Preview uses a data URL; upload uses a Blob captured from the same canvas.
  const preview = document.getElementById("snapPreview");
  preview.src = canvas.toDataURL("image/jpeg", 0.9);
  preview.classList.add("show");

  const analyzeCamBtn = document.getElementById("analyzeCamBtn");
  analyzeCamBtn.disabled = true; // re-enabled once the blob is ready, just below
  canvas.toBlob(
    (blob) => {
      capturedImageBlob = blob;
      analyzeCamBtn.disabled = false;
    },
    "image/jpeg",
    0.9
  );

  if (cameraStream) {
    cameraStream.getTracks().forEach((t) => t.stop());
    cameraStream = null;
  }
  video.style.display = "none";
  document.getElementById("camOverlay").style.display = "none";
  document.getElementById("camStatus").style.display = "none";
}

// ---------------------------------------------------------------------
// Analyze
// ---------------------------------------------------------------------
async function analyzeImage(source) {
  const blob = source === "upload" ? uploadedImageFile : capturedImageBlob;
  if (!blob) return;

  hideResults();
  document.getElementById("loadingCard").classList.add("show");
  const btnId = "analyze" + (source === "upload" ? "Upload" : "Cam") + "Btn";
  document.getElementById(btnId).disabled = true;

  try {
    const apiResult = await window.CropLensAPI.identify(blob);
    const viewModel = window.CropLensAPI.toViewModel(apiResult);
    renderResult(viewModel);
  } catch (err) {
    document.getElementById("loadingCard").classList.remove("show");
    document.getElementById("errorCard").classList.add("show");
    document.getElementById("errorMsg").textContent =
      err.message || "Failed to analyse. Please try again.";
  }
  document.getElementById(btnId).disabled = false;
}

// ---------------------------------------------------------------------
// Render result
// ---------------------------------------------------------------------

// Hides a content section AND the ".section-divider" label immediately
// above it, so a section with nothing to show doesn't leave an orphaned
// heading (e.g. "Growth Stages" with an empty box under it).
function toggleSectionWithDivider(sectionEl, visible) {
  if (!sectionEl) return;
  sectionEl.style.display = visible ? "" : "none";
  const divider = sectionEl.previousElementSibling;
  if (divider && divider.classList.contains("section-divider")) {
    divider.style.display = visible ? "" : "none";
  }
}

function renderResult(r) {
  document.getElementById("loadingCard").classList.remove("show");
  document.getElementById("resultCard").classList.add("show");
  document.getElementById("resetBtn").style.display = "flex";

  document.getElementById("cropEmoji").textContent = r.emoji || "🌱";
  document.getElementById("cropName").textContent = r.cropName || "Unknown Plant";
  document.getElementById("sciName").textContent = r.sciName || "";

  setTimeout(() => {
    document.getElementById("confidenceFill").style.width = (r.confidence || 0) + "%";
  }, 80);
  document.getElementById("confidenceLabel").textContent = r.lowConfidence
    ? "Not confident — try a clearer photo"
    : (r.confidence || 0) + "% confidence";

  document.getElementById("growingSeason").textContent = r.growingSeason || "—";
  document.getElementById("waterNeeds").textContent = r.waterNeeds || "—";
  document.getElementById("timeNeeded").textContent = r.timeNeeded || "—";
  document.getElementById("temperature").textContent = r.temperature || "—";

  // Fertilizer section: the API gives a textual N-P-K recommendation
  // (crop_info.json's "npk" field), not a numeric split, so the bars
  // only render when r.npk is explicitly provided (currently never, by
  // design - see api.js's toViewModel).
  const fertilizerSection = document.querySelector(".fertilizer-section");
  const npkLegend = document.querySelector(".npk-legend");
  const npkBars = document.querySelector(".npk-bars");
  toggleSectionWithDivider(fertilizerSection, !!r.fertRec);
  if (r.npk) {
    npkLegend.style.display = "";
    npkBars.style.display = "";
    setTimeout(() => {
      document.getElementById("nBar").style.width = r.npk.n + "%";
      document.getElementById("pBar").style.width = r.npk.p + "%";
      document.getElementById("kBar").style.width = r.npk.k + "%";
    }, 150);
    document.getElementById("nVal").textContent = r.npk.n + "%";
    document.getElementById("pVal").textContent = r.npk.p + "%";
    document.getElementById("kVal").textContent = r.npk.k + "%";
  } else {
    npkLegend.style.display = "none";
    npkBars.style.display = "none";
  }
  document.getElementById("fertRec").textContent = r.fertRec || "—";

  // Growing-conditions grid: only shown once at least 2 fields are present.
  const careItems = [
    { label: "Soil Type", value: r.soilType, icon: "🪱" },
    { label: "Sunlight", value: r.sunlight, icon: "☀️" },
    { label: "Spacing", value: r.spacing, icon: "📐" },
    { label: "Soil pH", value: r.ph, icon: "🧫" },
  ].filter((i) => i.value);
  const careSection = document.getElementById("careSection");
  if (careItems.length >= 2) {
    document.getElementById("careGrid").innerHTML = careItems
      .map(
        (i) =>
          '<div class="care-item"><strong>' +
          i.icon +
          " " +
          i.label +
          "</strong>" +
          i.value +
          "</div>"
      )
      .join("");
    careSection.classList.add("show");
  } else {
    careSection.classList.remove("show");
  }

  // Growth timeline: hidden entirely when there's nothing to plot.
  const timelineSection = document.querySelector(".timeline-section");
  const timelineContainer = document.getElementById("timelineSteps");
  timelineContainer.innerHTML = "";
  (r.timeline || []).forEach((step) => {
    const div = document.createElement("div");
    div.className = "t-step";
    div.innerHTML =
      '<div class="t-dot">' +
      step.emoji +
      '</div><div class="t-label">' +
      step.label +
      '</div><div class="t-dur">' +
      step.dur +
      "</div>";
    timelineContainer.appendChild(div);
  });
  toggleSectionWithDivider(timelineSection, (r.timeline || []).length > 0);

  // Medicinal profile
  const medSection = document.getElementById("medicinalSection");
  if (r.hasMedicinal) {
    medSection.style.display = "block";
    document.getElementById("medTags").innerHTML = (r.medicinalTags || [])
      .map((t) => '<span class="med-tag">' + t + "</span>")
      .join("");
    document.getElementById("medDesc").textContent = r.medicinalDesc || "—";
  } else {
    medSection.style.display = "none";
  }

  document.getElementById("resultCard").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------
function resetAll() {
  uploadedImageFile = null;
  capturedImageBlob = null;
  if (cameraStream) {
    cameraStream.getTracks().forEach((t) => t.stop());
    cameraStream = null;
  }
  document.getElementById("previewImg").classList.remove("show");
  document.getElementById("uploadZone").style.display = "block";
  document.getElementById("fileInput").value = "";
  document.getElementById("analyzeUploadBtn").disabled = true;
  document.getElementById("videoEl").style.display = "none";
  document.getElementById("camOverlay").style.display = "none";
  document.getElementById("camStatus").style.display = "flex";
  document.getElementById("camStatus").innerHTML =
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="opacity:0.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg> Camera not started yet';
  document.getElementById("snapPreview").classList.remove("show");
  document.getElementById("snapBtn").disabled = true;
  document.getElementById("analyzeCamBtn").disabled = true;
  document.getElementById("startCamBtn").disabled = false;
  hideResults();
  window.scrollTo({ top: 0, behavior: "smooth" });
}
