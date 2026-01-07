---
# try also 'default' to start simple
theme: seriph
background: media/OzakiCoverPhoto.jpg
# some information about your slides (markdown enabled)
title: OzakiStencil
class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true
layout: cover
# duration of the presentation
# duration: 35min
---

<div style="background-color: rgba(180, 180, 180, 0.2); margin-top:-14em;" class="p-4">

# OzakiStencil 

## Emulating High-Precision Stencil Computations on INT8 Tensor Cores


</div>

<div class="abs-bl m-6 text-xl">
    <pre><code>Lund University, in collaboration with RIKEN R-CCS</code></pre>
</div>

<div class="abs-br m-6 text-xl">
  <button @click="$slidev.nav.openInEditor()" title="Open in Editor" class="slidev-icon-btn">
    <carbon:edit />
  </button>
  <a href="https://github.com/slidevjs/slidev" target="_blank" class="slidev-icon-btn">
    <carbon:logo-github />
  </a>
</div>

---
title: RIKEN Organization
layout: image-left
image: https://static01.nyt.com/images/2020/06/22/business/22supercomputer/merlin_173572923_346e2662-290d-4d66-a388-c779bb824123-mediumSquareAt3X.jpg
level: 1
transition: slide-up
---

# R-CCS in RIKEN

- Home to the Fugaku Supercomputer. 
- Organized into two _units_, **Science of Computing** and **Science by Computing**.
- _Units_ composed of _teams_ (5-year projects).

<br>
<div v-click="1">
<img src="./media/RCCS.png">
<div v-mark.circle.orange="2" class="jens"></div>
</div>

<style>
.jens {
    position: absolute;
    width: 6em;
    height: 6em;
    right: 13.7em;
    top: 23em;
    z-index: 0;
}
</style>

---
title: SPR Team Research
layout: two-cols-header
level: 1
transition: slide-left
---

# Supercomputing Performance Research Team

What will go into FugakuNext?

::left::

- SPR will inform FugakuNext's hardware.
- Research on what supercomputers spend most of their time in.
- Should RIKEN invest in TCs and AI Accelerators?

<img src="./media/SPR_Result.png">

::right::

### The Need for High Precision

- SPR and "Science by Computing" teams notice FP64 performance is stagnating.

<img src="./media/GPU_Precision_Trends.png">

<style>
.two-cols-header {
  column-gap: 4em;
}
</style>

<br>

- Looking for ways to use low-precision (or emulate high-precision) for their applications.

---
title: The Ozaki Scheme
layout: two-cols-header
level: 1
transition: slide-left
---

# The Ozaki I Scheme
Invented by Katsuhisa Ozaki in RIKEN 

::left::

- High-precision matrix multiplication using low-precision arithmetic.

Say we want to emulate FP64 operations using INT8.



$$
\mathcal{A} = \{3.0, 0.3\times10^3, 0.3\times10^{-3}\}
$$


