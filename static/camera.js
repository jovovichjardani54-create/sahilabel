/*
 * Browser camera capture for the labelled SahiLabel upload slots.
 * This module only owns media streams; index.html owns upload state and the
 * existing /check form submission flow.
 */
(() => {
  let stream = null;
  let activeLabel = null;

  const dialog = () => document.getElementById("cameraDialog");
  const video = () => document.getElementById("cameraVideo");
  const message = () => document.getElementById("cameraMessage");

  function setMessage(text) {
    const element = message();
    if (element) element.textContent = text || "";
  }

  function stop() {
    if (stream) stream.getTracks().forEach((track) => track.stop());
    stream = null;
    const preview = video();
    if (preview) preview.srcObject = null;
  }

  function close() {
    stop();
    const modal = dialog();
    if (modal?.open) modal.close();
    activeLabel = null;
    setMessage("");
  }

  async function open(label) {
    const modal = dialog();
    if (!modal) return;
    close();
    activeLabel = label;
    modal.showModal();
    if (!navigator.mediaDevices?.getUserMedia) {
      setMessage("Live camera is unavailable in this browser. Use Mobile capture instead.");
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
      const preview = video();
      preview.srcObject = stream;
      await preview.play();
    } catch (error) {
      setMessage("Could not access the camera. Allow permission, or use Mobile capture.");
      stop();
    }
  }

  function capture() {
    const preview = video();
    if (!stream || !preview?.videoWidth || !activeLabel) {
      setMessage("Start the camera before capturing an image.");
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = preview.videoWidth;
    canvas.height = preview.videoHeight;
    canvas.getContext("2d").drawImage(preview, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob || !activeLabel) {
        setMessage("The camera image could not be captured. Please try again.");
        return;
      }
      const file = new File([blob], `camera-${activeLabel}-${Date.now()}.jpg`, {
        type: "image/jpeg",
      });
      window.sahiLabelSelectImage?.(activeLabel, file);
      close();
    }, "image/jpeg", 0.92);
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-camera-open]").forEach((button) => {
      button.addEventListener("click", () => open(button.dataset.cameraOpen));
    });
    document.getElementById("cameraCapture")?.addEventListener("click", capture);
    document.getElementById("cameraCancel")?.addEventListener("click", close);
    dialog()?.addEventListener("close", stop);
  });
  window.addEventListener("beforeunload", stop);
  window.sahiLabelCamera = { stop, close };
})();
