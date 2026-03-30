Got it mrkj — here’s the **full, thorough** website + product web outline that includes **Ark language + compiler + docs**, and the **ArkNet multi-part system** (dev ↔ registry ↔ providers), with **exact pages**, **what each must contain**, and **subsystems needed so the pages are real**.

---

## 0) Top-level IA (information architecture)

You’re shipping **one brand** with **three pillars**:

1. **Ark Language** (syntax, semantics, stdlib, examples)
2. **Toolchain** (compiler, runtime, debugger, package/project tooling)
3. **ArkNet** (registry + providers + remote execution + billing)

So the website splits into:

* **Public marketing site**
* **Docs portal**
* **Authenticated consoles**
* **Status / Trust / Legal**

---

## 1) Public marketing site pages (conversion + credibility)

### 1.1 Home (/)

**Primary goal:** convert both sides: devs + providers.

**Above-the-fold (must contain)**

* Hero: “Write once, run local or ArkNet.”
* CTAs: **Get started (Dev)** / **Become a provider**
* 3–6 live indicators (no fluff):

  * Warm dispatch latency **P50/P95**
  * Cold start P95
  * Success rate last 24h
  * Active GPU capacity (tiered)
  * Median price per compute unit
  * Regions online

**Sections**

* “What Ark is” (language) + “What ArkNet is” (network)
* How it works (3-step):

  1. `arkc` compiles kernels + metadata
  2. runtime routes by `runtime{...}`
  3. registry resolves providers, provider executes, result returns
* Demo section: show 1 kernel + `Ark.toml` + `.env` snippet
* Trust section: “compiler-truth arg layout”, “budget enforcement”, “audit trails”
* Use cases: image ops, simulations, inference, sweeps
* Pricing teaser: credits + enterprise subscription + provider earnings
* Footer: Docs / Status / Security / Terms / Contact / GitHub (if public)

---

### 1.2 Language landing (/language)

**Goal:** “This is a real language, not a toy.”

* What makes Ark different (first-class kernels, async model, config/env schemas)
* Syntax tour:

  * `fn[cpu]`, `fn[gpu]`, `fn[host]`
  * `allocof<T>(n)`
  * `A <- kernel(A, ...) as t1; await t1;`
  * `schema { meta { format: env/toml } }`
  * `runtime{...}` + `@runtime preset`
* Semantics overview:

  * pointwise kernel meaning
  * memory model (tracked allocations, remote safe subset)
  * determinism and safety rules for remote eligibility
* Link to docs: “Language spec”, “Stdlib”, “Examples”, “FAQ”

---

### 1.3 Compiler & toolchain landing (/toolchain)

**Goal:** “Fast path from code → run.”

* `arkc` overview:

  * compile
  * run locally
  * run remote via runtime preset
* Targets:

  * `localhost`, `cpu`, `gpu:0`, `runtimes.arknet.io`, `provider_hash`
* Build pipeline narrative:

  * AST → MIR → LLVM/MLIR lowering → artifact + metadata → runtime
* Tooling:

  * formatter, linter, package/project tooling (if planned)
  * debug + trace integration (show it)
* Roadmap section: GPU remote once artifact backend is real

---

### 1.4 ArkNet landing (/arknet)

**Goal:** explain the network without handwaving.

* What ArkNet is: registry + provider daemon + developer runtime
* “Feels local” promise (measurable):

  * warm dispatch target, cache hit ratio target
* Key features:

  * persistent provider connection
  * kernel UID caching
  * buffer semantics (IDs + deltas later)
  * budgets and caps enforced client-side + server-side
* Link to Registry marketplace, Provider onboarding, Dev console

---

### 1.5 Registry marketplace (public) (/registry)

**Goal:** transparency + trust + adoption loop.

* Search/filter providers:

  * GPU model, VRAM, CPU/RAM
  * region, latency estimate, uptime score
  * price, queue depth
  * “verified” badges
* Provider cards (must show):

  * provider hash (copy)
  * specs
  * health score, success rate, last-seen
  * pricing
  * “Copy Ark.toml snippet”
* Pools view:

  * “recommended”
  * “lowest cost”
  * “lowest latency”
  * “highest reliability”
* Public registry stats:

  * active providers
  * capacity by tier
  * incident banner (if degraded)

---

### 1.6 Providers landing (/providers)

**Goal:** convert operators.

* Value prop: earn by contributing GPU/CPU with controls
* Install: provider agent download + quick start
* Controls:

  * % allocation, schedules
  * max VRAM, max RAM, max burn
  * allowlist/denylist projects (enterprise later)
* Safety posture:

  * sandbox roadmap, syscall restrictions roadmap
  * signed binaries, auth model
* Earnings calculator:

  * by GPU tier + uptime
* Provider requirements:

  * uptime minimum
  * penalties model (clarity prevents disputes)

---

### 1.7 Developers landing (/developers)

**Goal:** “ship today”.

* Quickstart snippet:

  * `.env` + `Ark.toml` + runtime preset
* Running modes:

  * local CPU
  * local GPU
  * remote provider hash
  * remote pool
* “Costs and controls”

  * max burn
  * timeout
  * fallback rules
* Debugging:

  * trace IDs
  * rerun locally
  * inspect artifact cache hits

---

### 1.8 Pricing (/pricing)

Two-sided pricing page:

* **Indie devs**

  * free tier + starter credits
  * credit packs (transparent)
* **Enterprise**

  * subscription + SLA + invoice billing
  * team features + RBAC + audit logs
* **Providers**

  * payout model
  * fee %
  * payout schedule and thresholds

---

### 1.9 Company pages

* /security (trust model + disclosure)
* /status (public incident + uptime)
* /about
* /careers
* /terms, /privacy, /acceptable-use

---

## 2) Docs portal (must be first-class)

Docs are split into **Language**, **Compiler/Runtime**, **ArkNet**.

### 2.1 Docs home (/docs)

* Quick start (5 minutes)
* Choose your path:

  * “I’m writing Ark code”
  * “I’m integrating Ark into a system”
  * “I’m becoming a provider”
* Search + version selector

---

### 2.2 Language docs (/docs/language)

**Must include**

* Syntax & types
* Functions & domains: host/cpu/gpu
* Memory and allocation:

  * `allocof<T>(n)` semantics
  * ownership and free (if exposed)
* Kernel semantics:

  * pointwise mapping and grid semantics
  * allowed operations
* Async semantics:

  * launch, await, detach, sync_all
* Schemas:

  * `schema Env meta{format: env}`
  * `schema Cfg meta{format: toml, path, section}`
  * presence semantics: `or`, `require`
* Runtime presets:

  * `runtime{...}` meaning
  * `@runtime preset` meaning
* Standard library reference
* Examples cookbook:

  * image ops
  * reductions (if supported)
  * multi-kernel pipelines
* Error model:

  * compile-time errors
  * runtime errors
  * remote errors

---

### 2.3 Compiler docs (/docs/compiler)

**Must include**

* `arkc` CLI reference:

  * compile
  * run
  * target selection
* Pipeline overview:

  * AST → MIR → lowering → artifact
* Kernel metadata:

  * Arg layout contract
  * UID definition
  * remote eligibility rules
* Build outputs:

  * where artifacts go
  * debugging outputs
* Versioning:

  * protocol version
  * ABI rules

---

### 2.4 Runtime docs (/docs/runtime)

**Must include**

* Runtime API surface (public):

  * async functions
  * allocator tracking
  * registration and launch semantics
* Local runtimes:

  * cpu pool
  * gpu device
* Remote runtime:

  * provider resolution
  * cache semantics
  * pointer/len semantics
  * budgets + timeouts
* Observability:

  * trace IDs
  * logs
  * metrics exports

---

### 2.5 ArkNet docs (/docs/arknet)

**Must include**

* Registry API (public and auth’d)
* Provider agent:

  * installation
  * config
  * updates
  * telemetry
* Provider protocol:

  * framing header
  * submit/complete
  * missing artifact retry model (if used)
* Billing model:

  * credits ledger
  * burn caps
  * dispute process
* Security model:

  * today
  * roadmap

---

## 3) Authenticated consoles (product web app)

### 3.1 Developer Console (/app)

**Pages**

1. Dashboard

   * spend today/month
   * warm P50/P95
   * success rate
   * cache hit rate
2. Projects

   * tokens, `.env` templates
   * `Ark.toml` generator
3. Runtimes & presets

   * saved targets
   * budget policies
4. Runs / Jobs

   * list view + trace view
   * provider used, cost, logs, artifact hit/miss
5. Billing

   * credits balance, invoices, budgets, alerts
6. Team (enterprise)

   * RBAC, audit logs

**Dev console subsystems**

* Auth + token issuance
* Usage metering pipeline
* Credits ledger + payments
* Run records + trace store
* Budget enforcement + alerts
* Artifact cache introspection

---

### 3.2 Provider Console (/provider)

**Pages**

1. Setup

   * register machine
   * obtain provider hash + keys
2. Live status

   * utilization, queue depth
   * success rate, disconnects
3. Policies

   * capacity %, schedules
   * max VRAM/RAM
4. Earnings

   * earnings/day
   * payout schedule
5. Reputation

   * score breakdown
   * reasons for penalties

**Provider subsystems**

* Provider registration + verification
* Heartbeats + telemetry ingestion
* Reputation scoring engine
* Payout ledger + risk controls

---

### 3.3 Admin Console (/admin)

**Must exist even v1**

* Provider moderation (ban/quarantine)
* Incident kill-switches
* Pricing controls (floors/ceilings)
* Fraud detection dashboards
* Dispute resolution workflow

---

## 4) “Realness” subsystems checklist (so pages aren’t marketing lies)

### Identity & auth

* Accounts + sessions
* API tokens (dev runtime token, provider agent token)
* RBAC + audit logs

### Registry core

* provider directory:

  * provider_hash → endpoints, pubkey, specs, pricing, status
* ranking / selection strategies
* search index + filtering

### Dispatch + accounting

* request tracing (cookie/trace ID)
* metering events
* credits ledger and burning logic
* budgets:

  * per-run cap
  * daily cap
  * timeout enforcement

### Artifact + caching coordination

* kernel UID storage
* per-provider “uploaded kernels set”
* cache hit tracking

### Telemetry + reliability

* provider heartbeat ingestion
* health scoring:

  * timeouts
  * disconnect rate
  * completion rate
* “last seen” and availability windows

### Payments

* dev payments (card, invoice)
* provider payouts (thresholds, holdback for new providers)
* disputes and refunds

### Observability

* log store
* trace store
* customer-facing status API

### Support / ops

* incident management + status publishing
* rate limiting + abuse prevention
* compliance baseline (enterprise)

---

## 5) What the landing / providers / registry must show (non-negotiable indicators)

### Landing

* warm latency P50/P95
* success rate 24h
* active capacity by GPU tier
* median cost
* regions online
* cache hit ratio

### Providers page

* estimated earnings by tier
* uptime requirement + penalty model
* time-to-first-earn median
* safety model summary + roadmap

### Registry page

* providers online count
* per-provider reliability score + last 24h failure rate
* price + queue depth
* copy-paste runtime snippet

---

## 6) Minimum v1 scope that still looks enterprise-grade

Ship v1 with:

* Home, Language, Toolchain, ArkNet, Registry, Devs, Providers, Pricing, Docs, Security, Status
* Dev Console: dashboard + runs + billing + tokens
* Provider Console: setup + status + payouts

Defer:

* complex pool strategies
* heavy benchmarking
* deep compliance pages (but don’t ignore security narrative)

---

**Suggestion:** make the docs + registry the “truth anchors” (versioned contracts, real metrics, copy-paste config snippets). If those two feel rock-solid, new devs trust Ark fast.
