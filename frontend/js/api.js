// All communication with the CropLens API lives here. app.js never calls
// fetch() directly - it calls window.CropLensAPI.identify()/toViewModel().
//
// The real API (see api/app/schemas.py) returns a lean shape:
//   { label, confidence (0-1), low_confidence, top_k, care, model_version }
// This UI was originally built against a richer mock shape (sciName,
// timeline, medicinal profile, numeric N-P-K, soil pH, etc). toViewModel()
// bridges the two: it fills what the real API provides and explicitly
// nulls out what it doesn't, rather than inventing data. app.js then hides
// the sections that have nothing to show - see its toggleSection() calls.

(function () {
  "use strict";

  const API_BASE_URL = (window.CROPLENS_CONFIG && window.CROPLENS_CONFIG.API_BASE_URL) || "";

  // Matches the "Optimised for" chips in index.html and models/v1/labels.json.
  const CLASS_INFO = {
    black_cumin: { displayName: "Black Cumin", sciName: "Nigella sativa", emoji: "🌱" },
    sweet_pea: { displayName: "Sweet Pea", sciName: "Lathyrus odoratus", emoji: "🌸" },
    onion: { displayName: "Onion", sciName: "Allium cepa", emoji: "🧅" },
  };

  function classInfo(label) {
    if (CLASS_INFO[label]) return CLASS_INFO[label];
    // Fallback for any class not in the table above (e.g. after retraining
    // with more classes and forgetting to update this list).
    const pretty = label
      ? label.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
      : "Unknown Plant";
    return { displayName: pretty, sciName: "", emoji: "🌿" };
  }

  /**
   * Upload an image and get a raw API prediction back.
   * @param {Blob} blob - a File (from <input type=file>) or a Blob
   *   (from canvas.toBlob() for camera captures). Both work with FormData.
   * @returns {Promise<object>} the API's PredictResponse JSON.
   * @throws {Error} with a user-friendly message on any failure.
   */
  async function identify(blob) {
    if (!blob) throw new Error("No image to analyse.");

    const formData = new FormData();
    formData.append("file", blob, "photo.jpg");

    let response;
    try {
      response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
      });
    } catch (networkErr) {
      throw new Error("Could not reach the CropLens API. Check your connection and try again.");
    }

    let data;
    try {
      data = await response.json();
    } catch (parseErr) {
      throw new Error(`Unexpected response from the server (status ${response.status}).`);
    }

    if (!response.ok) {
      // FastAPI's HTTPException responses are {"detail": "..."} - see
      // api/app/main.py and app/schemas.py's ErrorResponse.
      throw new Error(data.detail || `Server error (${response.status}).`);
    }

    return data;
  }

  /**
   * Adapt the API's PredictResponse into the shape index.html's
   * renderResult() expects. Fields the API doesn't provide are explicitly
   * null (never fabricated) - app.js hides the corresponding UI sections.
   */
  function toViewModel(apiResult) {
    const info = classInfo(apiResult.label);
    const care = apiResult.care || null; // null when low_confidence, or class not in crop_info.json

    return {
      lowConfidence: !!apiResult.low_confidence,
      emoji: info.emoji,
      cropName: info.displayName,
      sciName: info.sciName,
      confidence: Math.round((apiResult.confidence || 0) * 100),

      growingSeason: care ? care.season : null,
      waterNeeds: care ? care.water : null,
      timeNeeded: care ? care.harvest : null,
      temperature: null, // not collected by crop_info.json

      npk: null, // no numeric N-P-K split available - see fertRec instead
      fertRec: care ? care.npk : null, // textual fertilizer guidance

      soilType: null,
      sunlight: null,
      spacing: null,
      ph: null, // none of these four are in crop_info.json yet

      timeline: [], // not collected by crop_info.json

      hasMedicinal: false, // not collected by crop_info.json
      medicinalTags: [],
      medicinalDesc: null,

      topK: apiResult.top_k || [],
      modelVersion: apiResult.model_version,
    };
  }

  window.CropLensAPI = { identify, toViewModel, classInfo };
})();
