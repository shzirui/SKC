<template>
  <main class="main">
    <section class="section header">
      <div class="venue">Anonymous submission</div>
      <h1 class="title">Selective Knowledge Control</h1>
      <div class="subtitle">
        Continual Learning of GUI Agents Over Application Streams
      </div>

      <div class="author-list">
        <span class="author">Zirui Shang<sup>1,2</sup>, Xin Shu<sup>3</sup>, Yang Liu<sup>2</sup>, Zhi Gao<sup>1,2,4</sup>, Xinxiao Wu<sup>1,4</sup>, Lifeng Fan<sup>2</sup></span>
      </div>
      <div class="affiliation-list">
        <div>1. Beijing Key Laboratory of Intelligent Information Technology, School of Computer Science &amp; Technology, Beijing Institute of Technology</div>
        <div>2. State Key Laboratory of General Artificial Intelligence, BIGAI</div>
        <div>3. Wuhan University</div>
        <div>4. Guangdong Laboratory of Machine Perception and Intelligent Computing, Shenzhen MSU-BIT University</div>
        <div class="corresponding">Corresponding author: gaozhibit@bit.edu.cn</div>
      </div>

      <div class="links">
        <a class="link-button disabled" href="#" aria-disabled="true">
          <i class="fas fa-file-pdf"></i>
          <span>Paper</span>
        </a>
        <a class="link-button" href="https://github.com/shzirui/SKC" target="_blank">
          <i class="fab fa-github"></i>
          <span>Code</span>
        </a>
        <a class="link-button" href="https://huggingface.co/XinShu3047/gui-agent-checkpoints" target="_blank">
          <i class="fa fa-robot"></i>
          <span>Model</span>
        </a>
      </div>
    </section>

    <section class="tldr">
      <b>TL;DR</b>
      We introduce activation-conditioned selective knowledge control, a lightweight
      continual-learning method that protects application-specific GUI knowledge while
      allowing shared GUI skills to remain adaptable.
    </section>

    <section class="section">
      <img class="hero-figure" src="/skc/shared_specific.png" alt="Shared and application-specific GUI knowledge">
      <p class="caption">
        GUI applications share reusable procedures, such as opening menus and export dialogs,
        while also requiring application-specific options and workflows.
      </p>
    </section>

    <section class="section">
      <h2 class="section-title">Introduction</h2>
      <p class="intro">
        GUI agents powered by multimodal large language models need to keep adapting as new
        applications, layouts, and workflows appear. This naturally creates an application-stream
        continual-learning problem: the agent must learn the current application without
        forgetting previous ones.
      </p>
      <p class="intro">
        Existing preservation strategies often treat historical knowledge uniformly. That is too
        coarse for GUI agents. Some historical knowledge is application-specific and should be
        protected from unnecessary updates, while other knowledge is shared across applications and
        should remain trainable. SKC makes this distinction dynamically through neuron activations.
      </p>
    </section>

    <section class="section">
      <h2 class="section-title">Method</h2>
      <img class="wide-figure" src="/skc/framework.png" alt="Selective Knowledge Control framework">
      <div class="method-grid">
        <div class="method-item">
          <div class="method-index">1</div>
          <h3>Historical State</h3>
          <p>
            SKC stores a compact state over protected MLP neurons and their historical update
            directions after each application stage.
          </p>
        </div>
        <div class="method-item">
          <div class="method-index">2</div>
          <h3>Activation Partition</h3>
          <p>
            During new-application training, forward hooks identify which protected neurons are
            activated by the current batch.
          </p>
        </div>
        <div class="method-item">
          <div class="method-index">3</div>
          <h3>Selective Gradient Control</h3>
          <p>
            Inactive protected neurons are truncated, while activated protected neurons are
            projected orthogonal to historical update directions.
          </p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2 class="section-title">Experiments</h2>
      <p class="intro">
        We report the main quantitative results in the same order as the paper: Table 1 gives the
        overall comparison, while Tables 3 and 6 highlight the more detailed breakdowns.
      </p>
      <div class="experiment-stack">
        <figure class="experiment-item full-width">
          <img class="experiment-table" src="/skc/table1.png" alt="Table 1">
          <figcaption>Table 1: Main comparison.</figcaption>
        </figure>
        <div class="experiment-row">
          <figure class="experiment-item half-width">
            <img class="experiment-table" src="/skc/table3.png" alt="Table 3">
            <figcaption>Table 3: Detailed result.</figcaption>
          </figure>
          <figure class="experiment-item half-width">
            <img class="experiment-table" src="/skc/table6.png" alt="Table 6">
            <figcaption>Table 6: Additional result.</figcaption>
          </figure>
        </div>
      </div>
    </section>

    <section class="section">
      <h2 class="section-title">Visualization</h2>
      <div class="image-row">
        <div>
          <img class="result-figure" src="/skc/offset_heatmap.png" alt="Historical update direction heatmap">
          <p class="caption">Historical update directions provide the subspaces used for projection.</p>
        </div>
        <div>
          <img class="result-figure" src="/skc/knowledge_case_study.png" alt="Knowledge case study">
          <p class="caption">Case studies highlight shared and application-specific GUI behaviors.</p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2 class="section-title">Code Usage</h2>
      <p class="intro">
        The implementation is built on DART-GUI and verl. Enable SKC from the trainer launch script
        by setting the gradient-surgery switch and providing a historical state file.
      </p>
      <pre class="code-block"><code>gradient_surgery=True
gradient_surgery_state_path="/path/to/gradient_surgery_state.pt"
gradient_surgery_all_project=False
gradient_surgery_all_zero=False</code></pre>
      <p class="intro">
        Protection artifacts can be converted with <code>scripts/build_gradient_surgery_state.py</code>,
        while stage-wise states can be merged with <code>scripts/merge_gradient_surgery_state.py</code>.
      </p>
    </section>

    <section class="section" id="BibTeX">
      <div class="bibtex-header">
        <h2 class="section-title left">BibTeX</h2>
        <button class="copy-button" :class="{ copied: copySuccess }" @click="copyBibtex">
          <i class="fas" :class="copySuccess ? 'fa-check' : 'fa-copy'"></i>
          {{ copySuccess ? 'Copied' : 'Copy' }}
        </button>
      </div>
      <pre class="bibtex-container"><code>{{ bibtexText }}</code></pre>
    </section>

    <footer class="footer">
      This website is inspired by DART-GUI, TongUI, MathVista, and Nerfies.
    </footer>
  </main>
</template>

<script setup>
import { ref } from 'vue'

const copySuccess = ref(false)

const bibtexText = `@misc{skc2027,
  title = {Selective Knowledge Control for Continual Learning of GUI Agents Over Application Streams},
  author = {Anonymous Authors},
  year = {2027},
  note = {Anonymous submission}
}`

const copyBibtex = async () => {
  try {
    await navigator.clipboard.writeText(bibtexText)
    copySuccess.value = true
    setTimeout(() => {
      copySuccess.value = false
    }, 2000)
  } catch (err) {
    copySuccess.value = false
  }
}
</script>

<style scoped>
.main {
  width: min(1160px, calc(100% - 32px));
  margin: 0 auto;
  color: #202635;
}

.section {
  margin: 54px 0;
  text-align: center;
}

.header {
  margin-top: 58px;
}

.venue {
  display: inline-block;
  margin-bottom: 14px;
  padding: 6px 12px;
  border: 1px solid #d9deea;
  border-radius: 999px;
  color: #536078;
  background: #ffffff;
  font-size: 0.92rem;
}

.title {
  margin: 0;
  font-size: clamp(3rem, 8vw, 5.8rem);
  line-height: 0.98;
  font-weight: 800;
}

.subtitle {
  width: min(100%, 900px);
  margin: 20px auto 0;
  color: #4f5c72;
  font-size: clamp(1.55rem, 3.4vw, 2.55rem);
  line-height: 1.15;
}

.author-list {
  margin-top: 22px;
  color: #536078;
  font-size: 1.08rem;
}

.affiliation-list {
  width: min(100%, 920px);
  margin: 14px auto 0;
  color: #536078;
  font-size: 0.98rem;
  line-height: 1.55;
}

.corresponding {
  margin-top: 8px;
  color: #2d3d56;
  font-weight: 650;
}

.links {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 12px;
  margin-top: 26px;
}

.link-button {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 40px;
  padding: 0 18px;
  border-radius: 999px;
  color: #ffffff;
  background: #222a37;
  text-decoration: none;
  font-weight: 650;
}

.link-button:hover {
  background: #3a4659;
}

.link-button.disabled {
  cursor: not-allowed;
  opacity: 0.62;
}

.tldr {
  max-width: 980px;
  margin: 42px auto 10px;
  padding: 18px 22px;
  border-left: 4px solid #2663ff;
  background: #ffffff;
  box-shadow: 0 8px 28px rgba(31, 42, 68, 0.08);
  text-align: left;
  font-size: 1.1rem;
  line-height: 1.6;
}

.section-title {
  margin: 0 0 20px;
  font-size: clamp(1.8rem, 4vw, 2.4rem);
  line-height: 1.16;
}

.section-title.left {
  text-align: left;
}

.intro {
  max-width: 980px;
  margin: 14px auto;
  text-align: justify;
  color: #334057;
  font-size: 1.04rem;
  line-height: 1.75;
}

.hero-figure,
.wide-figure {
  display: block;
  width: 100%;
  max-width: 980px;
  margin: 0 auto;
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 10px 32px rgba(31, 42, 68, 0.1);
}

.wide-figure {
  max-width: 1060px;
}

.caption {
  max-width: 860px;
  margin: 14px auto 0;
  color: #59657a;
  font-size: 0.96rem;
  line-height: 1.55;
}

.method-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
  margin-top: 22px;
}

.method-item {
  min-height: 220px;
  padding: 22px;
  border: 1px solid #dfe5f0;
  border-radius: 8px;
  background: #ffffff;
  text-align: left;
  box-shadow: 0 6px 20px rgba(31, 42, 68, 0.06);
}

.method-index {
  width: 34px;
  height: 34px;
  border-radius: 999px;
  display: grid;
  place-items: center;
  color: #ffffff;
  background: #2663ff;
  font-weight: 800;
}

.method-item h3 {
  margin: 18px 0 10px;
  font-size: 1.22rem;
}

.method-item p {
  margin: 0;
  color: #48546a;
  line-height: 1.62;
}

.two-column {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(300px, 0.9fr);
  gap: 28px;
  align-items: center;
}

.side-figure {
  width: 100%;
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 8px 28px rgba(31, 42, 68, 0.08);
}

.stream {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  flex-wrap: wrap;
  margin: 8px 0 22px;
}

.stream-node {
  min-width: 150px;
  padding: 14px 16px;
  border: 1px solid #d7deeb;
  border-radius: 8px;
  background: #ffffff;
  font-weight: 750;
}

.stream-arrow {
  color: #2663ff;
  font-size: 1.6rem;
  font-weight: 800;
}

.image-row {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 22px;
  align-items: start;
}

.result-figure {
  width: 100%;
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 8px 26px rgba(31, 42, 68, 0.08);
}

.experiment-stack {
  max-width: 1060px;
  margin: 18px auto 0;
}

.experiment-item {
  margin: 0;
  padding: 0;
}

.experiment-item figcaption {
  margin-top: 10px;
  color: #59657a;
  font-size: 0.95rem;
  line-height: 1.45;
}

.experiment-table {
  display: block;
  width: 100%;
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 8px 26px rgba(31, 42, 68, 0.08);
}

.experiment-row {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 22px;
  margin-top: 22px;
  align-items: start;
}

.code-block,
.bibtex-container {
  width: 100%;
  max-width: 980px;
  margin: 18px auto 0;
  padding: 18px 20px;
  overflow-x: auto;
  border-radius: 8px;
  background: #1f2633;
  color: #f5f7fb;
  text-align: left;
  line-height: 1.55;
}

.bibtex-header {
  width: 100%;
  max-width: 980px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.copy-button {
  height: 36px;
  padding: 0 14px;
  border: 0;
  border-radius: 999px;
  color: #ffffff;
  background: #222a37;
  cursor: pointer;
  font-weight: 650;
}

.copy-button.copied {
  background: #1f7a4d;
}

.footer {
  margin: 90px 0 46px;
  color: #778197;
  text-align: center;
}

@media (max-width: 820px) {
  .main {
    width: min(100% - 20px, 1160px);
  }

  .section {
    margin: 40px 0;
  }

  .header {
    margin-top: 40px;
  }

  .title {
    font-size: clamp(2.65rem, 13vw, 3.35rem);
  }

  .subtitle {
    font-size: 1.42rem;
  }

  .tldr {
    width: 100%;
  }

  .method-grid,
  .two-column,
  .image-row {
    grid-template-columns: 1fr;
  }

  .intro {
    text-align: left;
  }

  .stream-arrow {
    transform: rotate(90deg);
  }
}
</style>
