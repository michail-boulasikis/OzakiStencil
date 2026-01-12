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

- RIKEN: Japan's largest research center.
- R-CCS: Kobe, home to the Fugaku SC. 
- Organized into two _units_, **Science of Computing** and **Science by Computing**.
- _Units_ composed of _teams_ (5-year projects).

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
- Should RIKEN/Japan invest in TCs and AI Accelerators?

<img v-click="1" src="./media/SPR_Result.png">

::right::

<div v-click="2">

### The Need for High Precision

- SPR and "Science by Computing" teams notice FP64 performance is stagnating.

<img src="./media/GPU_Precision_Trends.png">

<br>

- Looking for ways to use low-precision (or emulate high-precision) for their applications.

</div>

<style>
.two-cols-header {
  column-gap: 4em;
}
</style>

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

**Emulation Example:** Say we want to emulate operations on 3 FP32 numbers, $\alpha, \beta, \gamma$, using INT8.

<v-switch>
  <template #1> <img src="./media/Alpha_Beta_Gamma_1.svg"> </template>
  <template #2> <img src="./media/Alpha_Beta_Gamma_2.svg"> </template>
  <template #3-6> <img src="./media/Alpha_Beta_Gamma_3.svg"> </template>
</v-switch>

::right::

<div v-click="4">

- To compute $\alpha \times \beta$, we cross-multiply all slices...

<img width="80%" src="./media/AlphaBeta_Mul.svg">

</div>

<div v-click="5">

- ...and accumulate the partial products.

</div>

<style>
.two-cols-header {
  column-gap: 4em;
}
</style>

---
title: Ozaki Scheme Properties
level: 1
layout: two-cols-header
transition: slide-left
---

# Ozaki Scheme Properties
For any set of floating point numbers $\mathcal{A} \in \mathbb{F}^n$...

::left::

<div v-click="1">

- Higher dynamic range $\rightarrow$ more slices.
- Reduce slices by scaling rows and columns differently.

<img scale="85%" src="./media/OzakiScalingMatrices.svg">

- Emulation with `float` types is also possible.

</div>


<div v-click="2">

- Emulated `double` is <span v-mark.highlight.yellow="3">more accurate</span> than native `double` precision.

</div>

::right::

<div v-click="2">

**Example:** Compute $(10^{16} + 1) - 10^{16}$.
- Native arithmetic: loses the `+1.0` due to rounding. Result: `0.0` ❌
- Ozaki: Slices similar magnitude, carries propagated. Result: `1.0` ✅

</div>

<div v-click="4">

<br>
<hr>

Ozaki Scheme $\rightarrow$ solution to the precision problem in GPAI.

- But has only been used for matrix multiplication.
- Can we use it for more computation patterns?

</div>

---
title: OzakiStencil
level: 1
layout: two-cols
transition: slide-left
---

<img class="flavor" src="./media/PlusMinusBackground.jpg">

::right::

# OzakiStencil
Adapting the Ozaki Scheme for stencil loops.

<br>

<div v-click="1">

## Opportunities

- <span class="plus">GPU/AI Accelerator friendly</span> implementation.
- Can emulate <span class="plus">arbitrary/adaptive precision</span>.
- Allows for <span class="plus">coarser tiling</span>.

</div>

<br>

<div v-click="2">

## Challenges
- Needs to be <span class="minus">very efficient</span>.
  - <span class="minus">Overhead</span> much more noticeable.
- <span class="minus">Dynamic range changes</span> across loop iterations.

</div>

<style>
.two-cols-header {
  column-gap: 4em;
}
.plus {
  color: #6AB500;
}
.minus {
  color: #ED1313;
}
.flavor {
  position: absolute;
  top: -5em;
  right: 30em;
  width: 50%;
  height: auto;
}
</style>

---
title: OzStencil Efficiency
level: 1
layout: two-cols-header
transition: slide-left
---

# Improving OzakiStencil
Using stencil properties

::left::

Stencils operate on arrays (lattices)

<img class="px-6" style="width: 50%" src="./media/StencilArrayLattice.svg">

<div v-click="1">

Coefficients $c_i \in \alpha\mathbb{Z},\; \alpha \in \mathbb{R}$.
- Find $\alpha$ and factor it out.
- Resulting coefficients are integers.

</div>

::right::

<div v-click="2">

### Example (Heat Equation)

</div>
<v-switch>
  <template #2>

$\mathcal{S} = \left[ \frac{7}{16\pi^2} \; \frac{1 - \frac{14}{16}}{\pi^2} \; \frac{7}{16\pi^2}\right]$
  
  </template>
  <template #3-6>

$\mathcal{S} = \left[ \frac{7}{16\pi^2} \; \frac{1 - \frac{14}{16}}{\pi^2} \; \frac{7}{16\pi^2}\right] =$
    <span v-mark.highlight.yellow="4">
$\frac{1}{16\pi^2}\left[ 7 \quad 2 \quad 7\right]$
    </span>
  </template>
</v-switch>

<div v-click="4">

Can be represented with just one INT8 slice.

<img class="px-2" style="width: 75%" src="./media/StencilOneSlice.svg">

</div>

<br>

<div v-click="5">

- Useful with euclidean geometry and Taylor stencils.
- Not useful with spectral methods or unstructured meshes.

</div>

---
title: OzStencil Efficiency
level: 1
layout: two-cols-gap
transition: slide-left
gap: gap-12
---

# Improving OzakiStencil
Reducing overhead

Originally slicing and mending (reconstructing) every timestep.

- GEMM: Slicing & Mending <span class="plus">~5%</span> of runtime.
- Stencil: Slicing & Mending <span class="minus">~98%</span> of runtime.

<br>

<img v-click="1" src="./media/AmortizeOverhead.svg">

<br>

<div v-click="1">

- Can we <span v-mark.highlight.underline.green="1">keep the data sliced</span>?

</div>

::right::

<img v-click="2" style="width: 80%" src="./media/KeepResultSliced.svg">

<div v-click="3">

- Result needs to fit in current slices.
- Estimate when new slice is necessary (depends on stencil and boundary).
- Introduce mend & slice pair when needed.

</div>

<style>
.plus {
  color: #6AB500;
}
.minus {
  color: #ED1313;
}
</style>


---
title: OzStencil Efficiency
level: 1
layout: two-cols-gap
transition: slide-left
gap: gap-12
---


# What is Done

<div v-click="1">

- CPU versions and rudimentary GPU versions done.
- Confirmed that OzakiStencil works!

<img width="80%" src="./media/OzakiErrorsFigure.png">

</div>

::right::

# Left to Do

<div v-click="2">

- Optimize code to improve performance.
- Use INT8 tensor cores for integration.

<br>

<img src="./media/StencilAsGEMM.png">

<br>

- «Rigorize» slice and mend insertions.

</div>


---
title: Thank you
level: 1
class: text-center
layout: cover
background: ./media/ThankYou.jpg
---

<div style="background-color: rgba(180, 180, 180, 0.2); margin-top:-14em;" class="p-4">

# Thank you!

## Questions?

</div>
