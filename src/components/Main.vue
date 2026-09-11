<template>
  <main class="main">
    <section class="section header">
      <h1 class="title">Selective Knowledge Control</h1>
      <div class="subtitle">
        for Continual GUI Agents Learning Over Application Streams
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
        <a class="link-button" href="http://arxiv.org/abs/2609.06530" target="_blank">
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
      <div class="image-frame image-frame--hero">
        <img class="hero-figure" src="/skc/shared_specific.png" alt="Shared and application-specific GUI knowledge">
      </div>
      <p class="caption">
        GUI applications share reusable procedures, such as opening menus and export dialogs,
        while also requiring application-specific options and workflows.
      </p>
    </section>

    <section class="section">
      <h2 class="section-title">Introduction</h2>
      <p class="intro">
        Graphical User Interface (GUI) agents powered by Multimodal Large Language Models (MLLMs)
        have emerged as a pivotal paradigm for automating complex interactions on desktops or
        mobiles. Continual learning is a crucial capability for Graphical User Interface (GUI)
        agents to adapt to evolving applications while retaining knowledge acquired from previous
        applications. Such application streams pose a challenging knowledge modeling problem: new
        applications often share underlying knowledge with past ones, yet also introduce specific
        knowledge that must not interfere with historical knowledge. In this paper, we propose
        activation-conditioned selective knowledge control, a lightweight method that achieves
        selective knowledge retention via neuron-level gradient manipulation. Our method maintains
        a compact historical knowledge state to protect highly activated MLP neurons that preserve
        previous knowledge. When a new application arrives, it performs real-time gradient surgery
        conditioned on forward activation. Concretely, the protected neurons are categorized into
        two types: unactivated neurons holding specific knowledge, whose gradients are truncated to
        prevent interference; and activated neurons holding shared knowledge, whose gradients are
        orthogonally projected to preserve stability while enabling adaptation. After each
        application stage, newly identified critical neurons are merged into the historical state
        for future learning. Empirical evaluations on multi-app sequential benchmark demonstrate
        that our method effectively mitigates catastrophic forgetting on prior applications while
        sustaining robust adaptation to new ones.
      </p>
    </section>

    <section class="section">
      <h2 class="section-title">Method</h2>
      <div class="image-frame image-frame--wide">
        <img class="wide-figure" src="/skc/framework.png" alt="Selective Knowledge Control framework">
      </div>
      <div class="method-grid">
        <div class="method-item">
          <div class="method-index">01</div>
          <h3>Historical State</h3>
          <p>
            At the beginning of each stage, the method loads a historical knowledge state to
            protect highly activated MLP neurons with their historical update directions. The
            protected set tracks the neuron indices protected from past stages, while the
            corresponding historical update directions span the historical parameter-offset
            subspace for each neuron. After each application training, a state update scheme merges
            newly identified important neurons into the historical knowledge state for future
            learning.
          </p>
        </div>
        <div class="method-item">
          <div class="method-index">02</div>
          <h3>Activation Partition</h3>
          <p>
            During the forward phase of training on a new application, the proposed method registers
            forward hooks on the inputs of MLP down projections in Transformer blocks and computes
            a runtime activation score for each neuron. Rather than using a fixed numerical
            threshold, it selects the top activated neurons to form the current high-activation
            set. The protected historical neurons are then partitioned according to whether they
            are reused by the current update, assigning shared functional regions and
            application-specific historical regions.
          </p>
        </div>
        <div class="method-item">
          <div class="method-index">03</div>
          <h3>Selective Gradient Control</h3>
          <p>
            During the back-propagation phase, the proposed method modifies the gradients of the
            MLP projections at the neuron level before parameter updates. For protected historical
            neurons that are not activated by the current update, their gradients are truncated to
            prevent unnecessary interference with application-specific historical knowledge. For
            protected historical neurons that are activated by the current update, their gradients
            are projected onto the subspace orthogonal to cumulative historical update directions
            to preserve historical directions while allowing compatible adaptation. For unprotected
            neurons, the proposed method leaves the gradient unchanged.
          </p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2 class="section-title">Experiments</h2>
      <div class="experiment-stack">
        <div class="experiment-block">
          <h3 class="experiment-title">Main Results</h3>
          <p class="intro">
            Table 1 reports the average success rates over all applications observed up to each
            training stage. The proposed method improves the overall performance at every
            comparable stage from Stage 2 to Stage 8. The gains are especially clear in the middle
            and later stages, where the overall success rate increases by 12.6, 7.1, and 10.6
            percentage points at Stages 3, 6, and 7, respectively. These improvements indicate
            that activation-conditioned selective control helps the agent maintain stronger
            performance over the application stream instead of only adapting to the most recent
            application.
          </p>
          <figure class="experiment-item full-width">
            <div class="image-frame">
              <img class="experiment-table" src="/skc/table1.png" alt="Table 1">
            </div>
            <figcaption>Table 1: Average success rates over all applications observed up to each training stage.</figcaption>
          </figure>
        </div>

        <div class="experiment-block">
          <h3 class="experiment-title">Ablation Studies</h3>
          <p class="intro">
            We compare four update-control settings to isolate the effect of activation-conditioned
            partitioning and neuron-level gradient surgery. The naive fine-tuning baseline leaves
            gradients unchanged for all protected neurons. The static projection variant removes
            activation-conditioned partitioning and projects gradients for all protected neurons.
            The static freezing variant truncates gradients for all protected neurons. Our method
            applies activation-aware selective control to protected neurons. The baseline achieves
            a mean success rate of 37.6%. Static projection and static freezing improve the mean
            success rate to 41.5% and 40.3%, respectively. In contrast, our method reaches 47.1%
            on average, showing that activation-aware selective control better balances adaptation
            and preservation.
          </p>
          <div class="experiment-row">
            <figure class="experiment-item half-width">
              <div class="image-frame">
                <img class="experiment-table" src="/skc/table3.png" alt="Table 3">
              </div>
              <figcaption>Table 3: Ablation studies on selective control with four update-control settings.</figcaption>
            </figure>
            <figure class="experiment-item half-width">
              <div class="image-frame">
                <img class="experiment-table" src="/skc/table5.png" alt="Table 5">
              </div>
              <figcaption>Table 5: Hyperparameter analysis of the truncated SVD rank.</figcaption>
            </figure>
          </div>
          <p class="intro experiment-note">
            We analyze the effect of the truncated SVD rank used for historical direction storage.
            The SVD-rank table shows that r=4 achieves the best overall success rate of 42.7%,
            whereas r=2 and r=8 yield lower performance. These results indicate that the proposed
            method benefits from a moderate number of retained historical update directions.
          </p>
        </div>

        <div class="experiment-block">
          <h3 class="experiment-title">Training Overhead</h3>
          <p class="intro">
            We also measure the computational overhead introduced by activation-conditioned
            selective knowledge control during training. Table 6 compares the baseline and the
            proposed method under the same training configuration. Compared with the baseline, our
            method increases per-step training time from 4279.23s to 4324.00s, corresponding to a
            1.0% increase. Token throughput decreases by 3.3%, and compute throughput decreases by
            1.9%. These results support the lightweight design of the proposed method, which adds
            little per-step time cost while preserving most training throughput under the same
            training framework.
          </p>
          <figure class="experiment-item experiment-item--overhead">
            <div class="image-frame">
              <img class="experiment-table" src="/skc/table6.png" alt="Table 6">
            </div>
            <figcaption>Table 6: Training overhead evaluation. We report per-step training time, token throughput (Tokens/s), and compute throughput (TFLOPs) under the same training configuration.</figcaption>
          </figure>
        </div>
      </div>
    </section>

    <section class="section">
      <h2 class="section-title">Visualization</h2>
      <p class="intro visualization-description">
        We provide a qualitative case showing how shared procedural knowledge acquired at Stage 7
        is reused at Stage 8. During the Stage 7 VSCode task, our method learns a general procedure
        from the successful <em>Install from VSIX</em> workflow: open the installation interface,
        start local installation, select and confirm the source, and verify the installation. At
        Stage 8, this shared knowledge is applied to the Chrome task through application-specific
        controls: our method opens <em>Manage Extensions</em>, clicks <em>Load unpacked</em>, selects
        and confirms <em>helloExtension</em>, and verifies the extension card and success
        notification, thereby completing the task. In contrast, the baseline loses the confirmation
        control after the chooser changes state and terminates without confirming the source or
        verifying the installation, resulting in task failure. This case illustrates the transfer
        of shared procedural knowledge learned from the Stage 7 VSCode workflow to the Stage 8
        Chrome task.
      </p>
      <div class="visualization-case">
        <div class="image-frame">
          <img class="result-figure" src="/skc/knowledge_case_study.png" alt="Shared procedural knowledge reuse on a Chrome task">
        </div>
        <p class="caption">Shared procedural knowledge reuse on a Chrome task.</p>
      </div>
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

const bibtexText = ``

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
  margin: 72px 0;
  text-align: center;
}

.section + .section {
  padding-top: 10px;
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

.section-title::after {
  content: "";
  display: block;
  width: 44px;
  height: 3px;
  margin: 12px auto 0;
  border-radius: 3px;
  background: #2663ff;
}

.section-title.left {
  text-align: left;
}

.section-title.left::after {
  margin-left: 0;
}

.intro {
  max-width: 980px;
  margin: 14px auto;
  text-align: justify;
  color: #334057;
  font-size: 1.04rem;
  line-height: 1.75;
}

.image-frame {
  padding: 18px;
  border: 1px solid #dfe5f0;
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 8px 26px rgba(31, 42, 68, 0.08);
}

.image-frame--hero {
  width: 100%;
  max-width: 780px;
  margin: 0 auto;
}

.image-frame--wide {
  width: 100%;
  max-width: 1060px;
  margin: 0 auto;
}

.hero-figure,
.wide-figure {
  display: block;
  width: 100%;
  margin: 0;
  border-radius: 2px;
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
  width: 32px;
  height: 24px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #59657a;
  background: #f4f6fa;
  border: 1px solid #dfe5f0;
  border-radius: 4px;
  font-size: 0.8rem;
  font-weight: 700;
}

.method-item h3 {
  margin: 14px 0 10px;
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
  display: block;
  width: 100%;
  border-radius: 2px;
  background: #ffffff;
}

.visualization-description {
  margin-bottom: 22px;
}

.visualization-case {
  max-width: 900px;
  margin: 0 auto;
}

.experiment-stack {
  max-width: 1060px;
  margin: 18px auto 0;
}

.experiment-block + .experiment-block {
  margin-top: 52px;
  padding-top: 34px;
  border-top: 1px solid #dfe5f0;
}

.experiment-title {
  margin: 0 0 12px;
  color: #1f2a44;
  font-size: 1.42rem;
  line-height: 1.3;
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
  border-radius: 2px;
  background: #ffffff;
}

.experiment-row {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 22px;
  margin-top: 22px;
  align-items: stretch;
}

.experiment-row .experiment-item {
  display: grid;
  grid-template-rows: 1fr auto;
}

.experiment-row .image-frame {
  align-self: center;
}

.experiment-note {
  margin-top: 22px;
}

.experiment-item--overhead {
  max-width: 760px;
  margin: 0 auto;
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
    margin: 52px 0;
  }

  .section + .section {
    padding-top: 0;
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
