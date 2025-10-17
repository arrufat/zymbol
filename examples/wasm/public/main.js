const encoder = new TextEncoder();
const decoder = new TextDecoder();

async function loadWasm() {
  const response = await fetch('zymbol_wasm.wasm');
  if (!response.ok) throw new Error(`failed to fetch wasm: ${response.status}`);
  const bytes = await response.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports;
}

function writeBytes(exports, bytes) {
  if (bytes.length === 0) return { ptr: 0, len: 0 };
  const ptr = exports.zymbol_alloc(bytes.length);
  if (!ptr) throw new Error('WebAssembly allocator returned null');
  new Uint8Array(exports.memory.buffer, ptr, bytes.length).set(bytes);
  return { ptr, len: bytes.length };
}

function freeBytes(exports, ptr, len) {
  if (ptr && len) exports.zymbol_free(ptr, len);
}

function readResult(exports) {
  const len = exports.zymbol_result_len();
  if (len === 0) return '';
  const ptr = exports.zymbol_result_ptr();
  return decoder.decode(new Uint8Array(exports.memory.buffer, ptr, len));
}

function updateStatus(element, status, message) {
  element.dataset.status = status;
  element.textContent = message;
}

async function main() {
  const exports = await loadWasm();

  const exprInput = document.getElementById('expr');
  const varInput = document.getElementById('variable');
  const deriveButton = document.getElementById('derive');
  const resultPre = document.getElementById('result');
  const statusP = document.getElementById('status');
  const simplifyToggle = document.getElementById('simplify');
  if (!(simplifyToggle instanceof HTMLInputElement)) {
    throw new Error('missing simplify checkbox');
  }

  async function deriveExpression() {
    statusP.hidden = true;
    resultPre.dataset.status = 'ok';
    exports.zymbol_clear();

    const exprText = exprInput.value;
    const varText = varInput.value.trim() || 'x';

    let exprRef;
    let varRef;

    try {
      exprRef = writeBytes(exports, encoder.encode(exprText));
      varRef = writeBytes(exports, encoder.encode(varText));

      const status = exports.zymbol_derive(
        exprRef.ptr,
        exprRef.len,
        varRef.ptr,
        varRef.len,
        simplifyToggle.checked ? 1 : 0,
      );

      const output = readResult(exports);
      resultPre.textContent = output;
      if (status !== 0 || output.startsWith('error:')) {
        resultPre.dataset.status = 'error';
      }
    } catch (error) {
      resultPre.dataset.status = 'error';
      resultPre.textContent = error.message;
    } finally {
      if (exprRef) freeBytes(exports, exprRef.ptr, exprRef.len);
      if (varRef) freeBytes(exports, varRef.ptr, varRef.len);
    }
  }

  deriveButton.addEventListener('click', deriveExpression);
  simplifyToggle.addEventListener('change', deriveExpression);
  exprInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
      deriveExpression();
    }
  });

  await deriveExpression();
}

main().catch((error) => {
  const statusP = document.getElementById('status');
  statusP.hidden = false;
  updateStatus(statusP, 'error', error.message);
});
