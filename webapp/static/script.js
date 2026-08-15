const imageInput = document.getElementById("image-input");
const dropzone = document.getElementById("dropzone");
const dropzoneText = document.getElementById("dropzone-text");
const previewImg = document.getElementById("preview-img");
const form = document.getElementById("upload-form");
const statusEl = document.getElementById("status");
const analyzeBtn = document.getElementById("analyze-btn");
const resultsSection = document.getElementById("results");

dropzone.addEventListener("click", () => imageInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.style.borderColor = "#2563eb";
});
dropzone.addEventListener("dragleave", () => {
  dropzone.style.borderColor = "#475569";
});
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.style.borderColor = "#475569";
  if (e.dataTransfer.files.length) {
    imageInput.files = e.dataTransfer.files;
    showPreview(e.dataTransfer.files[0]);
  }
});

imageInput.addEventListener("change", () => {
  if (imageInput.files.length) showPreview(imageInput.files[0]);
});

function showPreview(file) {
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.style.display = "block";
  dropzoneText.style.display = "none";
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!imageInput.files.length) {
    statusEl.textContent = "Please choose an image first.";
    return;
  }

  analyzeBtn.disabled = true;
  statusEl.textContent = "Running TTA inference + Grad-CAM… this can take a few seconds.";
  resultsSection.style.display = "none";

  const formData = new FormData();
  formData.append("image", imageInput.files[0]);

  try {
    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      statusEl.textContent = "Error: " + (data.error || "prediction failed");
      analyzeBtn.disabled = false;
      return;
    }

    renderResults(data);
    statusEl.textContent = "";
  } catch (err) {
    statusEl.textContent = "Request failed: " + err.message;
  } finally {
    analyzeBtn.disabled = false;
  }
});

function renderResults(data) {
  document.getElementById("top-label").textContent = data.top_prediction;
  document.getElementById("top-prob").textContent =
    (data.top_probability * 100).toFixed(1) + "%";

  document.getElementById("orig-img").src = data.original_url;
  document.getElementById("heatmap-img").src = data.heatmap_url;
  document.getElementById("arrow-img").src = data.arrow_url;

  const barsContainer = document.getElementById("report-bars");
  barsContainer.innerHTML = "";
  data.report.forEach((item) => {
    const row = document.createElement("div");
    row.className = "bar-row";

    const label = document.createElement("span");
    label.textContent = item.label;

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill" + (item.positive ? " positive" : "");
    fill.style.width = (item.probability * 100).toFixed(1) + "%";
    track.appendChild(fill);

    const pct = document.createElement("span");
    pct.textContent = (item.probability * 100).toFixed(1) + "%";

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(pct);
    barsContainer.appendChild(row);
  });

  resultsSection.style.display = "block";
}
