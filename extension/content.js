console.log("Email Sensitivity Checker content script loaded. URL:", location.href);

// CONFIGURATION
let debounceTimer;
const DEBOUNCE_DELAY = 1000;
let isProcessing = false;
let lastCheckedText = "";


const SHIELD_ICON_SVG = `
<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
</svg>`;

const overlay = document.createElement('div');
overlay.id = 'sensitivity-overlay';
overlay.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    padding: 10px 15px;
    background: white;
    border: 1px solid #ccc;
    border-radius: 4px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    z-index: 2147483647; 
    font-family: 'Segoe UI', sans-serif;
    display: none;
    font-size: 13px;
    font-weight: 600;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    user-select: none;
`;

// Use innerHTML for SVG
const iconSpan = document.createElement('span');
iconSpan.style.display = 'flex';
iconSpan.innerHTML = SHIELD_ICON_SVG;
overlay.appendChild(iconSpan);

const textSpan = document.createElement('span');
overlay.appendChild(textSpan);

(document.body || document.documentElement).appendChild(overlay);

let currentHighlightIndex = -1;
overlay.addEventListener('click', () => {
    const highlights = document.querySelectorAll('.sensitivity-highlight');
    if (highlights.length === 0) return;
    currentHighlightIndex = (currentHighlightIndex + 1) % highlights.length;
    const target = highlights[currentHighlightIndex];
    if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'center' });
        target.style.transition = 'box-shadow 0.3s';
        target.style.boxShadow = '0 0 10px 4px rgba(255, 0, 0, 0.6)';
        setTimeout(() => target.style.boxShadow = '0 0 2px red', 1000);
    }
});

function updateOverlay(text, bgColor, textColor = "black") {
    overlay.style.display = 'flex';
    overlay.style.backgroundColor = bgColor;
    overlay.style.color = textColor;
    textSpan.textContent = text;
}

function hideOverlay() {
    overlay.style.display = 'none';
    lastCheckedText = ""; // Reset so re-opening same text triggers scan
    isProcessing = false;
}

// Highlighting
function getAllTextAndElements() {
    let textToScan = "";
    let elements = [];
    let isCompose = false;

    // Helper to check element and add to list
    const checkElement = (el) => {
        if (!el) return;
        // Check visibility (simple check)
        if (el.offsetParent === null && el.tagName !== 'BODY') return; 

        if (el.isContentEditable || el.tagName === 'TEXTAREA' || el.tagName === 'INPUT') {
            const val = el.innerText || el.value;
            // Only add if it has content or is a significant container
            if (val || el.isContentEditable) { 
                textToScan += val + "\n\n";
                elements.push(el);
                isCompose = true;
            }
        } else if (el.tagName === 'IFRAME') {
             try {
                const doc = el.contentDocument;
                if (doc && (doc.body.isContentEditable || doc.body.getAttribute('contenteditable') === 'true')) {
                    textToScan += doc.body.innerText + "\n\n";
                    elements.push(doc.body); 
                    isCompose = true;
                }
            } catch(e) {}
        }
    };

    const candidates = document.querySelectorAll('textarea, input[type="text"], [contenteditable="true"], iframe');
    candidates.forEach(checkElement);

    if (document.body && document.body.getAttribute('contenteditable') === 'true') {
        checkElement(document.body);
    }

    elements = [...new Set(elements)];

    return { text: textToScan, elements, isCompose };
}

function checkSensitivity() {
    
    const { text, elements, isCompose } = getAllTextAndElements();
    
    if (!isCompose || !text || text.trim().length < 2) {
        hideOverlay();
        return;
    }

    if (text === lastCheckedText) return;
    lastCheckedText = text;

    updateOverlay("Checking Draft...", "#f0f0f0");
    isProcessing = true;

    chrome.runtime.sendMessage(
        { action: "check_sensitivity", text: text },
        (response) => {
            isProcessing = false;
            const currentStatus = getAllTextAndElements();
            if (!currentStatus.isCompose) {
                hideOverlay();
                return;
            }

            if (chrome.runtime.lastError) {
                console.warn("Server connection error:", chrome.runtime.lastError.message);
                updateOverlay("Connection Error", "#fee2e2", "#991b1b");
                return;
            }

            if (response && response.success) {
                try {
                    const data = response.data;
                    
                    const sensitiveSentences = (data.sentences || []).filter(s => s.prediction === 1 && (s.confidence === undefined || s.confidence > 0.6));
                    
                    // Calculate Total PII Count 1 Sentence = 1 Sensitive Item
                    let totalPIICount = sensitiveSentences.length;

                    removeAllHighlights();

                    if (totalPIICount > 0) {
                        highlightSentencesMulti(sensitiveSentences, elements);

                        const msg = `Found (${totalPIICount}) sensitive items.`;
                        updateOverlay(msg, "#ffcccc", "#cc0000");
                    } else {
                        updateOverlay("Draft Safe.", "#ccffcc", "#006600");
                    }
                } catch (err) {
                    console.error("Error processing sensitivity data:", err);
                    updateOverlay("Client Error", "#fee2e2", "#991b1b");
                }
            } else {
                updateOverlay("Server Error", "#fff3cd", "#856404");
            }
        }
    );
}

function removeAllHighlights() {
    const clearFromRoot = (root) => {
        const highlights = root.querySelectorAll('.sensitivity-highlight');
        highlights.forEach(span => {
            const parent = span.parentNode;
            if (parent) {
                while (span.firstChild) {
                    parent.insertBefore(span.firstChild, span);
                }
                parent.removeChild(span);
                parent.normalize();
            }
        });
    };

    clearFromRoot(document);

    const frames = document.querySelectorAll('iframe');
    frames.forEach(iframe => {
        try {
            const doc = iframe.contentDocument;
            if (doc) clearFromRoot(doc);
        } catch(e) {}
    });
}

function highlightSentencesMulti(sentences, containers) {
    if (!containers || containers.length === 0) return 0;
    
    containers.forEach(container => {
        const doc = container.ownerDocument;
        if (doc && !doc.getElementById('sensitivity-styles')) {
            const style = doc.createElement('style');
            style.id = 'sensitivity-styles';
            style.textContent = `
                .sensitivity-highlight {
                    background-color: #ffcccc !important;
                    color: black !important;
                    border-bottom: 2px solid red !important;
                    box-shadow: 0 0 2px red !important;
                    text-decoration: none !important; 
                    display: inline !important;
                    visibility: visible !important;
                    opacity: 1 !important;
                }
            `;
            (doc.head || doc.body).appendChild(style);
        }
    });

    let totalHighlights = 0;
    sentences.forEach(sent => {
        if (sent.prediction === 1 && sent.text.length > 2) { 
            if (sent.highlight_spans && sent.highlight_spans.length > 0) {
                sent.highlight_spans.forEach(spanText => {
                    let highlighted = false;
                    containers.forEach(container => {
                         if(fuzzyHighlight(container, spanText)) highlighted = true;
                    });
                    if (highlighted) totalHighlights++;
                });
            } else {
                if (sent.text.length > 3) { 
                    let highlighted = false;
                    containers.forEach(container => {
                         if(fuzzyHighlight(container, sent.text)) highlighted = true;
                    });
                    if (highlighted) totalHighlights++;
                }
            }
        }
    });
    return totalHighlights;
}

function fuzzyHighlight(root, text) {
    const normalizedText = text.normalize('NFKC');
    // REMOVE PUNCTUATION for extremely robust matching
    const tokens = normalizedText.replace(/[^\p{L}\p{N}]+/gu, '').toLowerCase(); 
    if (!tokens) return false;

    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null, false);
    let node;
    const textNodes = [];

    while(node = walker.nextNode()) {
        if (node.parentElement.classList.contains('sensitivity-highlight')) continue;
        textNodes.push(node);
    }

    let fullString = "";
    const map = [];
    for (const n of textNodes) {
        const val = n.nodeValue.normalize('NFKC');
        for (let i = 0; i < val.length; i++) {
            const char = val[i];
            // Only map alphanumeric chars, skip punctuation/spaces
            if (/\p{L}|\p{N}/u.test(char)) {
                fullString += char.toLowerCase();
                map.push({ node: n, offset: i });
            }
        }
    }

    const matchCnt = tokens.length;
    let searchCursor = 0;
    let foundAny = false;
    let safety = 0;
    
    while (safety++ < 10000) { 
        const foundIdx = fullString.indexOf(tokens, searchCursor);
        if (foundIdx === -1) break;
        const startMap = map[foundIdx];
        const endMap = map[foundIdx + matchCnt - 1]; 

        if (startMap && endMap) {
            highlightRange(startMap.node, startMap.offset, endMap.node, endMap.offset + 1);
            foundAny = true;
        }
        searchCursor = foundIdx + matchCnt;
    }
    return foundAny;
}

function highlightRange(startNode, startOffset, endNode, endOffset) {
    try {
        const range = document.createRange();
        range.setStart(startNode, startOffset);
        range.setEnd(endNode, endOffset);
        const parent = range.commonAncestorContainer.parentElement;
        if (parent && parent.closest('.sensitivity-highlight')) return;

        const span = document.createElement('span');
        span.className = 'sensitivity-highlight';
        range.surroundContents(span);
    } catch (e) { }
}

document.addEventListener('input', (e) => {
    clearTimeout(debounceTimer);
    const target = e.target;
    let isEditable = false;
    if (target.isContentEditable || target.tagName === 'TEXTAREA' || target.tagName === 'INPUT') isEditable = true;
    if (!isEditable && target.tagName === 'BODY' && target.getAttribute('contenteditable') === 'true') isEditable = true;
    if (isEditable) {
        removeAllHighlights();
        debounceTimer = setTimeout(checkSensitivity, DEBOUNCE_DELAY);
    }
}, true);

document.addEventListener('paste', () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(checkSensitivity, DEBOUNCE_DELAY);
}, true);

const observer = new MutationObserver((mutations) => {
    const { isCompose } = getAllTextAndElements();
    
    if (!isCompose) {
        hideOverlay();
        clearTimeout(debounceTimer);
    } else {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(checkSensitivity, DEBOUNCE_DELAY);
    }
});

observer.observe(document.body, { childList: true, subtree: true });

console.log("Email Sensitivity: UI Polish Done.");
