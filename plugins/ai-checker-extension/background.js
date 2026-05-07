chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "checkAI",
    title: "Проверить на ИИ",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "checkAI") return;

  const selectedText = info.selectionText;
  if (!selectedText || selectedText.trim().length === 0) return;

  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: showLoader
  });

  try {
    const response = await fetch("http://103.76.52.244/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts: [selectedText], threshold: 0.001 })
    });

    if (!response.ok) throw new Error(`HTTP error: ${response.status}`);

    const data = await response.json();
    const result = data.results?.[0];

    if (!result) throw new Error("Пустой ответ от сервера");

    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: showResult,
      args: [result.label, result.score, selectedText]
    });

  } catch (err) {
    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: showError,
      args: [err.message]
    });
  }
});

function showLoader() {
  const existing = document.getElementById("__ai_checker_popup__");
  if (existing) existing.remove();

  const popup = document.createElement("div");
  popup.id = "__ai_checker_popup__";
  popup.innerHTML = `
    <div class="aic-overlay" id="__aic_overlay__"></div>
    <div class="aic-popup">
      <div class="aic-spinner"></div>
      <div class="aic-loading-text">Анализируем текст…</div>
    </div>
  `;

  const style = document.createElement("style");
  style.id = "__ai_checker_style__";
  style.textContent = `
    #__ai_checker_popup__ * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
    .aic-overlay {
      position: fixed; inset: 0; background: rgba(0,0,0,0.45);
      z-index: 2147483646; backdrop-filter: blur(3px);
      animation: aicFadeIn 0.2s ease;
    }
    .aic-popup {
      position: fixed; top: 50%; left: 50%;
      transform: translate(-50%, -50%);
      z-index: 2147483647;
      background: #0f0f13;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 20px;
      padding: 40px 48px;
      min-width: 320px;
      max-width: 460px;
      text-align: center;
      box-shadow: 0 32px 80px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.04);
      animation: aicSlideIn 0.25s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    .aic-spinner {
      width: 44px; height: 44px;
      border: 3px solid rgba(255,255,255,0.08);
      border-top-color: #7c6aff;
      border-radius: 50%;
      margin: 0 auto 20px;
      animation: aicSpin 0.8s linear infinite;
    }
    .aic-loading-text { color: rgba(255,255,255,0.5); font-size: 14px; letter-spacing: 0.03em; }
    .aic-verdict {
      font-size: 13px; font-weight: 700; letter-spacing: 0.12em;
      text-transform: uppercase; margin-bottom: 8px; opacity: 0.6;
    }
    .aic-label {
      font-size: 52px; font-weight: 900; letter-spacing: -0.02em;
      line-height: 1; margin-bottom: 12px;
    }
    .aic-label.ai { color: #ff5c5c; }
    .aic-label.human { color: #4ade80; }
    .aic-score { font-size: 13px; color: rgba(255,255,255,0.35); margin-bottom: 24px; }
    .aic-excerpt {
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.07);
      border-radius: 10px;
      padding: 12px 16px;
      font-size: 12px;
      color: rgba(255,255,255,0.4);
      line-height: 1.6;
      text-align: left;
      margin-bottom: 24px;
      max-height: 72px;
      overflow: hidden;
      position: relative;
    }
    .aic-excerpt::after {
      content: '';
      position: absolute; bottom: 0; left: 0; right: 0; height: 28px;
      background: linear-gradient(transparent, rgba(15,15,19,0.95));
    }
    .aic-close {
      display: block; width: 100%;
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 10px;
      color: rgba(255,255,255,0.7);
      font-size: 14px;
      font-weight: 500;
      padding: 12px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .aic-close:hover { background: rgba(255,255,255,0.1); color: #fff; }
    .aic-error-icon { font-size: 36px; margin-bottom: 12px; }
    .aic-error-title { color: #ff5c5c; font-size: 16px; font-weight: 700; margin-bottom: 8px; }
    .aic-error-msg { color: rgba(255,255,255,0.4); font-size: 13px; margin-bottom: 24px; }
    @keyframes aicFadeIn { from { opacity: 0 } to { opacity: 1 } }
    @keyframes aicSlideIn { from { opacity: 0; transform: translate(-50%, -48%) scale(0.92) } to { opacity: 1; transform: translate(-50%, -50%) scale(1) } }
    @keyframes aicSpin { to { transform: rotate(360deg) } }
  `;

  const existingStyle = document.getElementById("__ai_checker_style__");
  if (existingStyle) existingStyle.remove();
  document.head.appendChild(style);
  document.body.appendChild(popup);

  document.getElementById("__aic_overlay__")?.addEventListener("click", () => {
    document.getElementById("__ai_checker_popup__")?.remove();
    document.getElementById("__ai_checker_style__")?.remove();
  });
}

function showResult(label, score, text) {
  const existing = document.getElementById("__ai_checker_popup__");
  if (existing) existing.remove();

  const isAI = label === "AI";
  const labelClass = isAI ? "ai" : "human";
  const verdict = isAI ? "Обнаружен ИИ-текст" : "Похоже на человека";
  const pct = score.toFixed(4);

  const popup = document.createElement("div");
  popup.id = "__ai_checker_popup__";
  popup.innerHTML = `
    <div class="aic-overlay" id="__aic_overlay__"></div>
    <div class="aic-popup">
      <div class="aic-verdict">${verdict}</div>
      <div class="aic-label ${labelClass}">${label}</div>
      <div class="aic-score">Score: ${pct}</div>
      <button class="aic-close">Закрыть</button>
    </div>
  `;

  document.body.appendChild(popup);

  const close = () => {
    document.getElementById("__ai_checker_popup__")?.remove();
    document.getElementById("__ai_checker_style__")?.remove();
  };
  popup.querySelector(".aic-close")?.addEventListener("click", close);
  document.getElementById("__aic_overlay__")?.addEventListener("click", close);
}

function showError(message) {
  const existing = document.getElementById("__ai_checker_popup__");
  if (existing) existing.remove();

  const popup = document.createElement("div");
  popup.id = "__ai_checker_popup__";
  popup.innerHTML = `
    <div class="aic-overlay" id="__aic_overlay__"></div>
    <div class="aic-popup">
      <div class="aic-error-icon">⚠️</div>
      <div class="aic-error-title">Ошибка запроса</div>
      <div class="aic-error-msg">${message}</div>
      <button class="aic-close">Закрыть</button>
    </div>
  `;

  document.body.appendChild(popup);

  const close = () => {
    document.getElementById("__ai_checker_popup__")?.remove();
    document.getElementById("__ai_checker_style__")?.remove();
  };
  popup.querySelector(".aic-close")?.addEventListener("click", close);
  document.getElementById("__aic_overlay__")?.addEventListener("click", close);
}
