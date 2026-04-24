<script>
  /** @type {HTMLInputElement} */
  let fileInput;
  let previewUrl = '';
  let selectedFile = null;
  let loading = false;
  let error = '';
  let result = null;
  let dragging = false;

  const LABEL_COLORS = { apple: '#ef4444', banana: '#eab308', orange: '#f97316' };

  /** @param {string} label */
  function labelColor(label) {
    return LABEL_COLORS[label] ?? '#6b7280';
  }

  /** @param {File | null | undefined} file */
  function setFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    selectedFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    result = null;
    error = '';
  }

  /** @param {Event} e */
  function onFileChange(e) {
    setFile(/** @type {HTMLInputElement} */ (e.target).files?.[0]);
  }

  /** @param {DragEvent} e */
  function onDrop(e) {
    e.preventDefault();
    dragging = false;
    setFile(e.dataTransfer?.files?.[0]);
  }

  async function detect() {
    if (!selectedFile) return;
    loading = true;
    error = '';
    result = null;

    try {
      const form = new FormData();
      form.append('file', selectedFile);
      const res = await fetch('/predict', { method: 'POST', body: form });
      if (!res.ok) throw new Error(`Server error ${res.status}: ${await res.text()}`);
      result = await res.json();
    } catch (/** @type {any} */ e) {
      error = e?.message ?? 'Unknown error';
    } finally {
      loading = false;
    }
  }
</script>

<main>
  <header>
    <h1>Fruit Detector</h1>
    <p>Upload a photo to detect apples, bananas, and oranges</p>
  </header>

  <!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
  <section
    class="upload-area"
    class:dragging
    role="region"
    aria-label="Image upload area"
    on:dragover|preventDefault={() => (dragging = true)}
    on:dragleave={() => (dragging = false)}
    on:drop={onDrop}
    on:click={() => fileInput.click()}
    on:keydown={(e) => e.key === 'Enter' && fileInput.click()}
  >
    {#if previewUrl}
      <img src={previewUrl} alt="Selected preview" class="preview" />
    {:else}
      <div class="placeholder">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
        <p>Drop an image here or <span class="link">click to browse</span></p>
      </div>
    {/if}
    <input bind:this={fileInput} type="file" accept="image/*" on:change={onFileChange} />
  </section>

  <button class="detect-btn" on:click={detect} disabled={!selectedFile || loading}>
    {#if loading}
      <span class="spinner" aria-hidden="true"></span> Detecting…
    {:else}
      Detect
    {/if}
  </button>

  {#if error}
    <div class="error" role="alert">{error}</div>
  {/if}

  {#if result}
    <section class="results">
      {#if result.image}
        <h2>Result</h2>
        <img
          src="data:image/jpeg;base64,{result.image}"
          alt="Annotated detections"
          class="annotated"
        />
      {:else}
        <div class="no-detections">No objects detected above the confidence threshold.</div>
      {/if}

      {#if result.detections?.length > 0}
        <table>
          <thead>
            <tr>
              <th>Class</th>
              <th>Confidence</th>
              <th>Bounding box (x1, y1, x2, y2)</th>
            </tr>
          </thead>
          <tbody>
            {#each result.detections as d}
              <tr>
                <td>
                  <span class="badge" style:background={labelColor(d.label)}>{d.label}</span>
                </td>
                <td>{(d.score * 100).toFixed(1)}%</td>
                <td class="box-coords">[{d.box.map((v) => v.toFixed(0)).join(', ')}]</td>
              </tr>
            {/each}
          </tbody>
        </table>
      {/if}
    </section>
  {/if}
</main>

<style>
  :global(*, *::before, *::after) {
    box-sizing: border-box;
  }
  :global(body) {
    margin: 0;
    font-family: system-ui, -apple-system, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    min-height: 100vh;
  }

  main {
    max-width: 760px;
    margin: 0 auto;
    padding: 2rem 1rem;
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
  }

  header h1 {
    margin: 0 0 0.25rem;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
  }
  header p {
    margin: 0;
    color: #94a3b8;
  }

  .upload-area {
    border: 2px dashed #334155;
    border-radius: 0.75rem;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    min-height: 220px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    outline: none;
  }
  .upload-area:hover,
  .upload-area.dragging {
    border-color: #6366f1;
    background: #1e293b;
  }
  .upload-area input {
    display: none;
  }

  .placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.75rem;
    color: #64748b;
    padding: 2rem;
    text-align: center;
  }
  .placeholder svg {
    width: 2.5rem;
    height: 2.5rem;
  }
  .placeholder p {
    margin: 0;
  }
  .link {
    color: #818cf8;
    text-decoration: underline;
  }

  .preview {
    max-width: 100%;
    max-height: 380px;
    object-fit: contain;
    border-radius: 0.5rem;
  }

  .detect-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.65rem 2rem;
    background: #6366f1;
    color: #fff;
    border: none;
    border-radius: 0.5rem;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    align-self: flex-start;
    transition: background 0.15s;
  }
  .detect-btn:hover:not(:disabled) {
    background: #4f46e5;
  }
  .detect-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .spinner {
    display: inline-block;
    width: 1rem;
    height: 1rem;
    border: 2px solid rgba(255, 255, 255, 0.4);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .error {
    background: #450a0a;
    border: 1px solid #7f1d1d;
    border-radius: 0.5rem;
    padding: 0.75rem 1rem;
    color: #fca5a5;
    font-size: 0.9rem;
  }

  .results {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  .results h2 {
    margin: 0;
    font-size: 1.1rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.8rem;
  }

  .annotated {
    max-width: 100%;
    border-radius: 0.5rem;
  }

  .no-detections {
    padding: 1.25rem;
    background: #1e293b;
    border-radius: 0.5rem;
    color: #94a3b8;
    text-align: center;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
  }
  th {
    text-align: left;
    padding: 0.5rem 0.75rem;
    background: #1e293b;
    color: #64748b;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
  }
  td {
    padding: 0.5rem 0.75rem;
    border-top: 1px solid #1e293b;
    vertical-align: middle;
  }

  .badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #fff;
    text-transform: capitalize;
  }

  .box-coords {
    color: #64748b;
    font-family: ui-monospace, monospace;
    font-size: 0.8rem;
  }
</style>
