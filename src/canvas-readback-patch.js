(() => {
  "use strict";

  if (!window.HTMLCanvasElement || HTMLCanvasElement.prototype.__facetraceReadbackPatch) {
    return;
  }

  const originalGetContext = HTMLCanvasElement.prototype.getContext;

  // The local detector, alignment, and quality paths read canvas pixels often.
  // Supplying this hint on first 2D-context creation avoids Chrome's repeated
  // getImageData readback warning and can improve CPU paths.
  HTMLCanvasElement.prototype.getContext = function patchedGetContext(contextId, options) {
    if (String(contextId).toLowerCase() !== "2d") {
      return originalGetContext.call(this, contextId, options);
    }

    const nextOptions = options && typeof options === "object"
      ? { ...options }
      : {};

    if (!Object.prototype.hasOwnProperty.call(nextOptions, "willReadFrequently")) {
      nextOptions.willReadFrequently = true;
    }

    return originalGetContext.call(this, contextId, nextOptions);
  };

  Object.defineProperty(HTMLCanvasElement.prototype, "__facetraceReadbackPatch", {
    value: true,
    configurable: false
  });
})();
