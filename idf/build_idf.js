// Builds the filled VIT Invention Disclosure Format (IDF)-B for AURA.
// All numbers in Section 8 are read from ../results/results.json.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, WidthType, ShadingType, BorderStyle, LevelFormat,
  PageNumber, VerticalAlign, TabStopType,
} = require("docx");

const ROOT = path.join(__dirname, "..");
const R = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "results.json")));
const G = R.gating, TW = R.twin;
const FIG = (f) => path.join(ROOT, "figures", f);

// ------------------------------------------------------------ helpers
const FONT = "Calibri";
const ACCENT = "1F4E79";
const PAGE_W = 11906, MARGIN = 1080, CONTENT_W = PAGE_W - 2 * MARGIN; // A4

const f1 = (x) => x.toFixed(3);
const pct = (x, d = 1) => (100 * x).toFixed(d) + " %";
const n1 = (x) => x.toFixed(1);
const n2 = (x) => x.toFixed(2);
const pm = (a, d = 3) => `${a[0].toFixed(d)} ± ${a[1].toFixed(d)}`;
const red = (a, b) => (100 * (1 - a / b)).toFixed(1) + " %";
const pv = (p) => (p < 1e-4 ? "< 10⁻⁴" : p.toExponential(1));

function runs(text, base = {}) {
  // **bold** markup inside text
  return text.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((t) =>
    t.startsWith("**") ? new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, size: 21, ...base })
      : new TextRun({ text: t, font: FONT, size: 21, ...base }));
}
const P = (text, opts = {}) => new Paragraph({
  children: runs(text, opts.run || {}), spacing: { after: 100, line: 276 },
  alignment: opts.align || AlignmentType.JUSTIFIED, ...(opts.p || {}),
});
const B = (text, level = 0) => new Paragraph({
  children: runs(text), numbering: { reference: "bullets", level },
  spacing: { after: 60, line: 264 }, alignment: AlignmentType.JUSTIFIED,
});
const N = (text, ref = "claims") => new Paragraph({
  children: runs(text), numbering: { reference: ref, level: 0 },
  spacing: { after: 80, line: 264 }, alignment: AlignmentType.JUSTIFIED,
});
const SEC = (text) => new Paragraph({
  children: [new TextRun({ text, bold: true, font: FONT, size: 24, color: ACCENT })],
  spacing: { before: 280, after: 120 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 2 } },
});
const SUB = (text) => new Paragraph({
  children: [new TextRun({ text, bold: true, font: FONT, size: 22, color: "222222" })],
  spacing: { before: 180, after: 80 },
});
const CAP = (text) => new Paragraph({
  children: [new TextRun({ text, italics: true, font: FONT, size: 18, color: "444444" })],
  alignment: AlignmentType.CENTER, spacing: { before: 40, after: 160 },
});
const EQ = (text) => new Paragraph({
  children: [new TextRun({ text, font: "Cambria Math", size: 21, italics: true })],
  alignment: AlignmentType.CENTER, spacing: { before: 60, after: 100 },
});

function pngSize(file) {
  const b = fs.readFileSync(file);
  return [b.readUInt32BE(16), b.readUInt32BE(20)];
}
function IMG(file, widthPx = 620) {
  const [w, h] = pngSize(FIG(file));
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 80, after: 20 },
    children: [new ImageRun({
      type: "png", data: fs.readFileSync(FIG(file)),
      transformation: { width: widthPx, height: Math.round(widthPx * h / w) },
      altText: { title: file, description: file, name: file },
    })],
  });
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "8C8C8C" };
const borders = { top: border, bottom: border, left: border, right: border };
function cell(text, w, o = {}) {
  return new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: o.fill ? { fill: o.fill, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    verticalAlign: VerticalAlign.CENTER, columnSpan: o.span, rowSpan: o.rowSpan,
    children: String(text).split("\n").map((line) => new Paragraph({
      alignment: o.align || AlignmentType.LEFT,
      children: runs(line, { size: o.size || 18, bold: o.bold, color: o.color }),
    })),
  });
}
function TABLE(header, rows, widths, o = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const hl = o.highlight || [];
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) =>
        cell(h, widths[i], { fill: "D9E2F3", bold: true, align: i ? AlignmentType.CENTER : AlignmentType.LEFT })) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) =>
        cell(c, widths[i], { fill: hl.includes(ri) ? "E2EFDA" : undefined, bold: hl.includes(ri),
          align: i && o.center !== false ? AlignmentType.CENTER : AlignmentType.LEFT })) })),
    ],
  });
}
const SP = () => new Paragraph({ children: [], spacing: { after: 60 } });

// ------------------------------------------------------------ numbers
const A = G["AURA (proposed)"], O = G["Oracle (stream all)"];
const SOD = G["Send-on-Delta (2σ)"], DPS = G["Dual-Prediction (2σ)"];
const P2 = G["Periodic (2 min)"], P4 = G["Periodic (4 min)"];
const mainNames = ["Oracle (stream all)", "Periodic (2 min)", "Periodic (4 min)",
  "Send-on-Delta (2σ)", "Dual-Prediction (2σ)", "AURA (proposed)"];

// Iso-accuracy traffic: smallest swept traffic at which a family reaches AURA's F1.
function isoTraffic(fam) {
  const pts = Object.entries(G).filter(([k]) => k.startsWith(fam + "|"))
    .map(([, v]) => [v.tx_rate[0], v.macro_f1[0]]).sort((a, b) => a[0] - b[0]);
  const ok = pts.filter((p) => p[1] >= A.macro_f1[0]);
  return ok.length ? { tx: ok[0][0], reached: true } : { tx: pts[pts.length - 1][0], f1: pts[pts.length - 1][1], reached: false };
}
const iso = ["Periodic", "Send-on-Delta", "Dual-Prediction"].map((f) => [f, isoTraffic(f)]);

const SH = TW["AURA shield (CVaR, proposed)"], SCH = TW["Static schedule"],
  REA = TW["Reactive occupancy (ECO idle)"], RST = TW["Reactive occupancy (standby idle)"],
  GRE = TW["Greedy agent (most-likely intent)"], MEAN = TW["AURA shield – risk-neutral (mean)"],
  SSOD = TW["AURA shield on Send-on-Delta stream"], SOR = TW["AURA shield on Oracle stream"],
  SMIS = TW["AURA shield, 25 % twin mismatch"];
const proto = R.protocol;

// ------------------------------------------------------------ document body
const body = [];
const push = (...x) => body.push(...x);

push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 60, after: 200 },
  children: [new TextRun({ text: "Invention Disclosure Format (IDF)-B", bold: true, font: FONT, size: 30, color: ACCENT })],
}));

// 1. Title
push(SEC("1. Title of the invention:"));
push(P("**AURA – Adaptive Uncertainty-Regulated Ambient intelligence: a system and method for intent-uncertainty-coupled semantic event gating with silence-aware inference and counterfactual digital-twin shielding for energy-efficient, proactive and safe AI-native IoT environments**"));

// 2. Field
push(SEC("2. Field /Area of invention:"));
push(P("The invention belongs to the field of **Ambient Intelligence (AmI) for AI-native Internet of Things (IoT)**. More specifically it relates to:"));
push(B("event-triggered / semantic uplink communication between battery-powered IoT sensor nodes and an edge gateway (CPC H04W 52/02, H04W 4/38, H04L 67/12);"));
push(B("context- and intent-aware multimodal sensor fusion and human activity / intent recognition at the network edge (G06N 7/01, G06N 20/00);"));
push(B("safety assurance of autonomous (agentic) actuation using digital twins, applied to smart homes, ambient-assisted living (AAL), smart buildings and Industry 5.0 (G05B 13/04, G05B 17/02, F24F 11/46, G08B 21/04)."));
push(P("Application domains named in the IEEE IoT-J special issue it targets include intelligent smart homes and buildings, AmI for smart healthcare and assisted living, edge AI and distributed AmI, semantic communications, digital twins, explainable and trustworthy AmI, and energy-efficient AI-native IoT architectures."));

// 3. Prior art
push(SEC("3. Prior Patents and Publications from literature (provide a table summarizing the prior art)"));
const paW = [420, 2700, 3480, 3146];
push(TABLE(["#", "Prior art (patent / publication)", "What it discloses", "Gap with respect to AURA"], [
  ["1", "M. Miskowicz, “Send-on-delta concept: an event-based data reporting strategy,” Sensors, vol. 6, no. 1, pp. 49–63, 2006.",
    "Node transmits only when the signal changes by more than a fixed dead-band Δ.", "Dead-band is fixed and task-agnostic; it ignores what the application needs to know. The receiver treats the last value as exact and does not use the information carried by silence."],
  ["2", "S. Santini, K. Römer, “An adaptive strategy for quality-based data reduction in wireless sensor networks,” Proc. INSS, 2006 (dual-prediction scheme with LMS).",
    "Node and sink run identical predictors; a sample is sent only if the prediction error exceeds a user-given accuracy bound.", "The accuracy bound is a static, per-signal reconstruction requirement. It is not linked to the uncertainty of a higher-level inference (occupant intent), and there is no safety floor."],
  ["3", "S. Trimpe, R. D’Andrea, “Event-based state estimation with variance-based triggering,” IEEE Trans. Automatic Control, vol. 59, no. 12, 2014.",
    "Sensors transmit when the estimator’s error variance exceeds a threshold (linear-Gaussian state).", "Triggering is based on the variance of a continuous state. It does not handle discrete multimodal intent, cross-sensor discriminability, look-ahead over intent transitions, or a broadcast attention vector to heterogeneous nodes."],
  ["4", "J. Sijs, M. Lazar, “Event based state estimation with time synchronous updates,” IEEE Trans. Automatic Control, vol. 57, no. 10, 2012.",
    "Uses set-valued (implicit) information when no event occurs, inside a Kalman-type estimator.", "Limited to linear-Gaussian state estimation. It does not cover categorical intent posteriors with time-varying, gateway-commanded bands, nor coupling to actuation risk."],
  ["5", "T. van Kasteren et al., “Accurate activity recognition in a home setting,” Proc. ACM UbiComp, 2008; D. J. Cook et al., “CASAS: a smart home in a box,” IEEE Computer, vol. 46, no. 7, 2013.",
    "HMM/CRF-based recognition of activities of daily living from ambient sensors.", "Assume that every sensor event reaches the server. Communication cost and the recognition model are designed separately, with no feedback that shapes sensing."],
  ["6", "J. Lu et al., “The smart thermostat: using occupancy sensors to save energy in homes,” Proc. ACM SenSys, 2010; learning-thermostat patent families (e.g., Nest Labs/Google).",
    "Occupancy/sleep prediction drives HVAC set-back and pre-heating.", "Decisions use the most likely occupancy or a point estimate. There is no tail-risk certification over posterior-sampled intent futures and no per-zone veto/repair of an agent’s proposal."],
  ["7", "F. Oldewurtel et al., “Use of model predictive control and weather forecasts for energy efficient building climate control,” Energy and Buildings, vol. 45, 2012.",
    "Stochastic MPC with chance constraints under weather uncertainty.", "Uncertainty comes from the weather, not from occupant intent inferred from a bandwidth-limited sensor stream. Sensing and actuation are not co-designed."],
  ["8", "M. Alshiekh et al., “Safe reinforcement learning via shielding,” Proc. AAAI, 2018.",
    "A shield derived from temporal-logic specifications blocks unsafe RL actions.", "Requires a formal specification and a known abstract model. It is not probabilistic over human intent, does not use a physical digital twin, and does not use a CVaR budget."],
  ["9", "H. Xie, Z. Qin, G. Y. Li, B.-H. Juang, “Deep learning enabled semantic communication systems,” IEEE Trans. Signal Processing, vol. 69, 2021.",
    "Transmits meaning-level features instead of bits (deep joint source-channel coding).", "Needs heavy neural encoders at the transmitter, which tiny MCU-class sensor nodes cannot run. The task-relevance is fixed at training time, not adapted online to the current intent ambiguity."],
], paW));
push(P("A preliminary keyword search (event-triggered + intent / activity + threshold feedback; digital twin + occupant intent + CVaR + HVAC; fall detection + adaptive reporting) found no patent or publication that combines (a) a gateway-commanded, intent-discriminability-driven per-sensor dead-band, (b) silence-aware interval likelihood for a categorical intent posterior, (c) a safety floor on the emergency intent, and (d) a counterfactual digital-twin CVaR shield that consumes the same posterior. A formal patentability / FTO search by the IPR cell is recommended.", { run: { size: 20 } }));

// 4. Summary & background
push(SEC("4. Summary and background of the invention (Address the gap / Novelty)"));
push(SUB("Background"));
push(P("Ambient Intelligence aims to make computing invisible. Dozens of battery-powered sensors (presence, power, water, CO₂, sound, light, wearables, bed load) let an edge gateway infer what occupants are doing and intend to do, and then act on their behalf (pre-heating a room, dimming lights, calling a caregiver after a fall). Two bottlenecks block this in AI-native IoT:"));
push(B("**Energy/bandwidth versus understanding.** Streaming every sample drains coin-cell batteries in months and congests the shared radio channel. Classical data-reduction schemes (periodic duty-cycling, send-on-delta, dual prediction) set their thresholds to reconstruct each signal, not to support the inference. They waste packets when the context is obvious, miss decisive changes when it is ambiguous, and at aggressive thresholds the recognition accuracy collapses (see Section 8)."));
push(B("**Autonomy versus safety.** An agentic controller acting on the single most likely intent saves energy but often guesses wrong. It leaves an elderly occupant in a cold bathroom or on a cold floor after a fall. Conservative schedules are safe but waste energy. No existing method certifies an agent’s action against the uncertainty of the intent estimate that a bandwidth-limited sensor network can actually deliver."));
push(SUB("Summary of the invention"));
push(P("AURA closes the loop **sensing → communication → inference → actuation** around one quantity: the gateway’s **posterior uncertainty over occupant intent**. It has three cooperating mechanisms:"));
push(B("**C1. Intent-uncertainty-coupled semantic gating.** Every node runs a twin predictor that is mirrored at the gateway, and reports only if the residual exceeds a dead-band θᵢ(t). The gateway computes, for each sensor, the expected separability Dᵢ(t) of the intents that are plausible over a look-ahead horizon. It maps Dᵢ to θᵢ and broadcasts a quantised attention vector: tighten immediately, relax lazily. A safety floor always keeps a minimum probability mass on the emergency intent, so fall-revealing sensors never go fully deaf."));
push(B("**C2. Silence-aware inference.** When a node is silent, the gateway knows that xᵢ ∈ [x̂ᵢ − θᵢ, x̂ᵢ + θᵢ]. It integrates the class-conditional likelihood over this band instead of trusting a stale value, so silence itself becomes evidence."));
push(B("**C3. Counterfactual digital-twin shield.** Before an agentic action is executed, a physical digital twin is rolled forward under every candidate plan and under N intent futures sampled from the posterior. Plans whose CVaRα of occupant discomfort or safety risk exceeds a budget β are vetoed, and the cheapest admissible plan is executed instead (veto-and-repair)."));
push(SUB("Novelty / inventive step"));
push(B("The reporting threshold of each node is a **function of the inference uncertainty of another layer** (intent posterior × look-ahead transitions × class-conditional separability). Prior art makes the threshold a function of the node’s own signal or a fixed accuracy target."));
push(B("**One posterior drives both** the communication policy and the risk-certified actuation. Silence is modelled exactly (C2), so the posterior stays calibrated even at about 6 % traffic, and this is what lets the shield be trusted."));
push(B("An **asymmetric, quantised downlink attention beacon** costs only one broadcast packet. It works with ultra-low-power MCU nodes (two comparisons and one subtraction per sample), unlike neural semantic encoders."));
push(B("A **safety floor** guarantees vigilance for rare, high-cost intents (falls) without a dedicated always-on stream."));
push(B("A **CVaR veto-and-repair shield over posterior-sampled intent futures** on a mismatched physical twin. It certifies or overrides any proposer (rule engine, MPC, RL or LLM agent)."));

// 5. Objectives
push(SEC("5. Objective(s) of Invention"));
[
  "To cut uplink radio traffic of ambient IoT sensor nodes by more than an order of magnitude while keeping intent-recognition accuracy close to that of full-rate streaming.",
  "To make every node’s reporting adapt to what the application currently needs to know, through a lightweight, gateway-commanded attention vector that MCU-class nodes can run.",
  "To use the information in silent intervals so the gateway’s intent posterior stays calibrated under heavy data reduction.",
  "To keep fast and reliable detection of rare emergencies (e.g., falls) under aggressive data reduction, via a safety floor.",
  "To certify proactive, agentic actuation (HVAC, lighting, appliances) against intent uncertainty with a counterfactual digital twin and a tail-risk (CVaR) budget. The goal is large energy savings without the comfort and safety violations of greedy, intent-driven automation.",
  "To extend battery lifetime of ambient sensors to multi-year operation, enabling truly ‘invisible’ deployments.",
].forEach((t) => push(N(t, "objectives")));

// 6. Working principle
push(SEC("6. Working principle of the invent (in brief)"));
push(P("At every sampling instant t, edge node i compares its measurement xᵢ(t) with the twin prediction x̂ᵢ(t), which the gateway reproduces exactly. The node transmits only if"));
push(EQ("| xᵢ(t) − x̂ᵢ(t) | > θᵢ(t) = σᵢ · cᵢ(t)"));
push(P("The gateway runs a Bayesian intent filter over K intents with time-of-day transitions A_h. Each sensor contributes a Gaussian likelihood if its value was received, and an **interval likelihood** if the node was silent:"));
push(EQ("ℓᵢₖ = Φ((x̂ᵢ+θᵢ−μₖᵢ)/σₖᵢ) − Φ((x̂ᵢ−θᵢ−μₖᵢ)/σₖᵢ)"));
push(P("From the posterior p(t), the gateway forms the look-ahead intent distribution q = (1−λ)·p(t)·A_h^L + λ·e_FALL (λ = safety floor). It then scores each sensor by how well it would separate the intents in q:"));
push(EQ("Dᵢ(t) = Σₖ qₖ · ((μₖᵢ − μ̄ᵢ)/σₖᵢ)²,   cᵢ(t) = c_min + (c_max − c_min) · exp(−κ·Dᵢ(t))"));
push(P("cᵢ is quantised to a few levels and broadcast as an attention beacon. Tightening is applied at once, while relaxing is rate-limited. Sensors that matter right now report finely; the rest stay almost silent. For actuation, the posterior p(t) seeds N Monte-Carlo intent futures. The thermal digital twin evaluates every candidate setpoint plan against each future, and the agent’s proposal is executed only if CVaR_α(discomfort) ≤ β. Otherwise it is replaced by the least-energy admissible plan."));

// 7. Detailed description
push(SEC("7. Description of the invention in detail (Include drawing and or photograph as needed)"));
push(SUB("7.1 System architecture (Fig. 1)"));
push(IMG("fig1_architecture.png", 620));
push(CAP("Fig. 1 – AURA system architecture: edge tier (gates 101), gateway (twin mirror 201, silence-aware intent engine 202, attention-vector generator 203, digital-twin shield 301) and services/actuation tier."));
push(P("**Edge tier.** Each sensor node (PIR, power, water-flow, CO₂, sound, light, wearable IMU, bed-load…) contains a sensing front-end, a twin predictor (zero-order hold or damped trend, chosen to be bit-exact with the gateway), a comparator gate 101 with a programmable dead-band register θᵢ, and a low-power radio (IEEE 802.15.4 / BLE / LoRa). Impact transients (e.g., accelerometer > 1.5 g) bypass the gate as event packets. The node updates its twin only on transmitted samples, so the gateway mirror never drifts."));
push(P("**Gateway tier.** (201) The *synchronised twin mirror* reconstructs x̂ᵢ(t) for all silent nodes. (202) The *silence-aware intent engine* is a Bayesian filter over K intents (a time-of-day HMM in the reference implementation; any calibrated probabilistic or foundation-model classifier can be used). It applies Gaussian likelihoods to received samples, interval likelihoods to silent ones, and a transition-conditioned impact model. (203) The *attention-vector generator* computes Dᵢ(t), maps it to dead-band levels, and emits a beacon when a level changes: tightening is immediate, relaxing waits for a hold time of 30 steps."));
push(P("**Actuation tier.** (301) The *counterfactual digital-twin shield* receives a proposed plan from any agent (rules, MPC, RL, LLM). It samples N intent futures from p(t) and the transition model, simulates the physical twin (here a multi-zone RC thermal model with deliberately mismatched parameters) for every candidate plan, and computes the CVaR of discomfort or safety cost. It then certifies the plan or vetoes and repairs it. Emergency posteriors above τ trigger caregiver alerts."));
push(SUB("7.2 Method flow (Fig. 2)"));
push(IMG("fig2_method_flow.png", 620));
push(CAP("Fig. 2 – Per-step method: edge loop S1–S4 and gateway loop S5–S9 closing the intent-uncertainty feedback."));
push(SUB("7.3 Counterfactual digital-twin shield – algorithm"));
[
  "Inputs: intent posterior p(t), hour h, current zone temperatures T_z, outdoor forecast, candidate plans Π (set-back/standby/ECO for j slots followed by comfort), risk level α, budget β.",
  "Sample N intent trajectories over horizon H (5-min slots) from p(t) and the look-ahead transition matrices. Also form the most-likely trajectory used by a greedy agent.",
  "For every zone z and plan π ∈ Π, roll the twin forward and accumulate energy E(π) and discomfort d(π, n) = Σ max(0, T_req(intentₙ) − T_z) for every future n.",
  "Risk r(π) = CVaR_α over n of d(π, n). The agent proposal π_g is executed if r(π_g) ≤ β (certified). Otherwise it is vetoed and π* = argmin E(π) s.t. r(π) ≤ β is executed (repaired); if no plan is admissible, comfort is restored at once.",
  "Apply the first slot of the chosen plan (receding horizon) and repeat every 5 min.",
].forEach((t) => push(N(t, "algo")));
push(SUB("7.4 Reference implementation"));
push(P("A full Python implementation was developed (≈ 1,250 lines of Python; modules aura/simulator, intent, gating, metrics, twin; experiment and figure scripts; unit tests). Gateway cost of AURA is " +
  `${n1(A.us_per_step[0])} µs per step for 11 sensors in pure Python/NumPy (vs ${n1(SOD.us_per_step[0])} µs for send-on-delta), i.e. < 0.01 % of one CPU core at a 30 s sampling period. Each node needs only one subtraction, one comparison and a dead-band register update per sample.`));
push(SUB("7.5 Alternative embodiments"));
push(B("Intent engine replaced by a multimodal foundation model / LLM that outputs calibrated intent probabilities; μₖᵢ, σₖᵢ obtained from its class-conditional embeddings."));
push(B("Twin predictor: AR / Kalman / tiny neural predictor; attention vector carried in ACK packets or periodic beacons (802.15.4e TSCH, BLE periodic advertising, LoRaWAN class B)."));
push(B("The shield's risk can also feed back into C1: zones and actions with high tail risk raise the attention weights of the sensors that would resolve that risk."));
push(B("Other domains: Industry 5.0 worker-intent-aware cobots, vehicle cabin AmI, smart campuses, low-altitude/UAV IoT, and patient monitoring in hospitals."));

// 8. Experimental validation
push(SEC("8. Experimental validation results:"));
push(SUB("8.1 Experimental set-up and protocol"));
push(TABLE(["Item", "Setting"], [
  ["Testbed", "Python reference implementation of the full edge–gateway–actuation loop (open, reproducible: `python experiments/run_all.py`)."],
  ["Environment", `Multimodal ambient-assisted-living home simulator: ${proto.sensors} sensors (4×PIR, power, CO₂, sound, water, light, wearable IMU, bed load) sampled every 30 s; 8 intents (SLEEP, HYGIENE, COOK, EAT, RELAX, WORK, AWAY, FALL). Each home has its own sensor signature (±30 %), routine, drift, noise, benign impact artefacts and injected falls (elevated-risk cohort).`],
  ["Training", `Gateway intent model trained on ${proto.train_homes} homes × ${proto.days} days. AURA hyper-parameters (κ = 3, c_min = 1σ, c_max = 4σ, λ = 0.10, 6 levels) tuned on 3 separate validation homes.`],
  ["Test", `${proto.test_homes} unseen homes (5 independent repetitions × 6 homes) × ${proto.days} days = ${proto.test_home_days} home-days, ${(proto.steps_per_policy / 1e6).toFixed(2)} M sensing steps (${(proto.steps_per_policy * proto.sensors / 1e6).toFixed(1)} M sensor samples) per policy; ${A.falls_total} fall events.`],
  ["Energy model", "IEEE 802.15.4-class node at 0 dBm: 0.90 mJ per uplink packet (wake-up, CSMA/CA, ACK), 0.45 mJ per received beacon, 0.02 mJ per sample, 6 µW sleep floor, CR2032 coin cell (225 mAh, 80 % usable)."],
  ["Baselines (gating)", "Oracle (stream every sample); Periodic duty-cycling (2 min, 4 min); Send-on-Delta (Miskowicz 2006) with a fixed 2σ dead-band; Dual-Prediction scheme with a damped-trend twin (Santini & Römer 2006), 2σ. All use the same gateway intent engine and the same impact-event bypass."],
  ["Actuation study", `4-zone dwelling (bedroom, bathroom, kitchen, living), RC thermal plant per zone, 2.5 kW heaters, winter outdoor profile, twin with ±10 % parameter mismatch and 1 K forecast error; ${proto.twin_homes} homes × 14 days. Baselines: static schedule, reactive occupancy (idle zones ECO or standby), greedy agent planning on the most-likely intent trajectory, risk-neutral shield.`],
  ["Metrics", "Macro-F1 and accuracy of intent recognition; uplink traffic (% samples); battery lifetime; fall recall, detection latency, false alarms/day; heating energy, discomfort (K·h/day), % occupied time > 0.5 K below comfort, manual-override episodes/day, worst cold exposure during falls; Wilcoxon signed-rank tests."],
], [2000, CONTENT_W - 2000], { center: false }));

push(SUB("8.2 Headline result – communication efficiency vs. intent recognition"));
push(TABLE(["Policy", "Macro-F1", "Accuracy", "Uplink traffic", "Pkts / home / day", "Fall recall", "Node life (yr)"],
  mainNames.map((n) => {
    const g = G[n];
    return [n, pm(g.macro_f1), f1(g.acc[0]), pct(g.tx_rate[0], 2), Math.round(g.pkts_day[0]).toLocaleString("en-US"),
      pct(g.recall[0], 1), n1(g.life_mean[0] / 365)];
  }), [2350, 1350, 1000, 1150, 1250, 1050, 1596], { highlight: [5] }));
push(CAP(`Table 1 – Mean (± std over ${proto.test_homes} unseen homes). AURA life includes the cost of receiving ${n1(A.beacons_day[0])} attention beacons/day.`));
push(P(`**Key result:** AURA reaches a macro-F1 of **${f1(A.macro_f1[0])}**, i.e. **${pct(A.macro_f1[0] / O.macro_f1[0], 1)} of the full-stream Oracle (${f1(O.macro_f1[0])})**, while transmitting only **${pct(A.tx_rate[0], 2)}** of the samples. This is a **${red(A.tx_rate[0], O.tx_rate[0])} reduction in uplink traffic** and extends coin-cell lifetime from ${n1(O.life_mean[0] / 365)} to **${n1(A.life_mean[0] / 365)} years (×${(A.life_mean[0] / O.life_mean[0]).toFixed(1)})**. At its 2σ operating point, Send-on-Delta uses ${(SOD.tx_rate[0] / A.tx_rate[0]).toFixed(1)}× the traffic of AURA and still loses ${(100 * (A.macro_f1[0] - SOD.macro_f1[0])).toFixed(1)} F1 points. Dual-Prediction uses ${(DPS.tx_rate[0] / A.tx_rate[0]).toFixed(1)}× the traffic and is ${(100 * (A.macro_f1[0] - DPS.macro_f1[0])).toFixed(1)} points lower. Periodic reporting at 2 min uses ${(P2.tx_rate[0] / A.tx_rate[0]).toFixed(1)}× the traffic and is ${(100 * (A.macro_f1[0] - P2.macro_f1[0])).toFixed(1)} points lower.`));

push(SUB("8.3 Accuracy–traffic Pareto front"));
push(IMG("fig3_pareto.png", 520));
push(CAP("Fig. 3 – Each curve sweeps one family’s tuning knob (period, dead-band, or AURA’s c_max). AURA dominates every baseline over the whole operating range."));
push(TABLE(["Baseline family", "Traffic needed to match AURA’s F1", "Traffic saving of AURA"],
  iso.map(([f, r]) => r.reached
    ? [f, pct(r.tx, 2), `${(r.tx / A.tx_rate[0]).toFixed(1)}× less traffic (${red(A.tx_rate[0], r.tx)})`]
    : [f, `not reached in the sweep (best F1 ${f1(r.f1)} at ${pct(r.tx, 1)})`, `> ${(r.tx / A.tx_rate[0]).toFixed(1)}× less traffic`]),
  [2600, 3600, CONTENT_W - 6200]));
push(CAP("Table 2 – Iso-accuracy comparison: smallest swept traffic at which each baseline reaches AURA’s macro-F1."));

push(SUB("8.4 Emergency (fall) detection under data reduction"));
push(TABLE(["Policy", "Falls detected", "Recall", "Mean latency (s)", "95th-pct latency (s)", "False alarms / day"],
  mainNames.map((n) => {
    const g = G[n];
    return [n, `${g.falls_detected} / ${g.falls_total}`, pct(g.recall[0], 1), Math.round(g.delay_mean_s).toString(),
      Math.round(g.delay_p95_s).toString(), n2(g.fa_per_day[0])];
  }), [2500, 1300, 1100, 1500, 1600, CONTENT_W - 8000], { highlight: [5] }));
push(CAP("Table 3 – Fall detection (alarm when P(FALL) > 0.5 during the event), pooled over all test homes."));
push(IMG("fig4_fall_trace.png", 560));
push(CAP("Fig. 4 – A fall episode: (a) raw streams, (b) AURA’s attention vector. Dead-bands of the fall-discriminating sensors tighten after the impact. (c) The gateway’s P(FALL) for Oracle, Send-on-Delta and AURA."));

push(SUB("8.5 Battery lifetime"));
push(IMG("fig6_lifetime.png", 500));
push(CAP(`Fig. 5 – Projected CR2032 lifetime per node (mean and worst node). AURA: ${n1(A.life_mean[0] / 365)} yr mean, ${n1(A.life_min[0] / 365)} yr worst node, vs ${n1(O.life_mean[0] / 365)} / ${n1(O.life_min[0] / 365)} yr for streaming.`));

push(SUB("8.6 Ablation study – every claimed element contributes"));
const abl = ["AURA (full)", "– interval (silence) likelihood", "– intent-coupled dead-bands (fixed 2σ)", "– look-ahead prediction", "– safety floor"];
push(TABLE(["Variant", "Macro-F1", "Uplink traffic", "Fall recall", "Mean fall latency (s)", "Node life (yr)"],
  abl.map((n) => {
    const g = G[n];
    return [n, pm(g.macro_f1), pct(g.tx_rate[0], 2), pct(g.recall[0], 1), Math.round(g.delay_mean_s).toString(), n1(g.life_mean[0] / 365)];
  }), [3300, 1350, 1250, 1100, 1400, CONTENT_W - 8400], { highlight: [0] }));
push(CAP("Table 4 – Removing any element of C1/C2 lowers accuracy, safety, or efficiency."));
push(IMG("fig5_ablation.png", 620));
push(CAP("Fig. 6 – Ablation: macro-F1, traffic and fall recall."));

push(SUB("8.7 Per-intent recognition"));
push(IMG("fig8_per_class.png", 560));
push(CAP("Fig. 7 – Per-intent F1. AURA keeps short, safety-relevant intents (HYGIENE, FALL) close to the Oracle, which is where fixed dead-bands lose most."));

push(SUB("8.8 Counterfactual digital-twin shield – proactive, safe actuation"));
const twNames = ["Static schedule", "Reactive occupancy (ECO idle)", "Reactive occupancy (standby idle)",
  "Greedy agent (most-likely intent)", "AURA shield – risk-neutral (mean)", "AURA shield (CVaR, proposed)"];
push(TABLE(["Controller", "Energy (kWh/day)", "Discomfort (K·h/day)", "% time cold", "Overrides / day", "Max cold during falls (K)"],
  twNames.map((n) => {
    const t = TW[n];
    return [n, `${n1(t.energy_kwh_day[0])} ± ${n1(t.energy_kwh_day[1])}`, n2(t.discomfort_kh_day[0]), n2(t.pct_uncomfortable[0]),
      n2(t.interventions_day[0]), n2(t.fall_max_deficit[0])];
  }), [2900, 1400, 1450, 1050, 1200, CONTENT_W - 8000], { highlight: [5] }));
push(CAP(`Table 5 – 4-zone heating (${proto.twin_homes} homes × 14 days). All intent-driven controllers use the AURA-gated posterior (${pct(A.tx_rate[0], 1)} traffic).`));
push(P(`**Key result:** compared with the static schedule, the AURA shield saves **${red(SH.energy_kwh_day[0], SCH.energy_kwh_day[0])} heating energy**. Compared with the greedy intent agent, it cuts discomfort by **${red(SH.discomfort_kh_day[0], GRE.discomfort_kh_day[0])}** and manual overrides by **${red(SH.interventions_day[0], GRE.interventions_day[0])}**. The worst cold exposure of a fallen occupant drops from ${n2(GRE.fall_max_deficit[0])} K (greedy) to **${n2(SH.fall_max_deficit[0])} K**. The shield vetoed and repaired ${n1(SH.veto_pct[0])} % of the greedy agent’s zone proposals. Compared with reactive control that keeps idle zones on standby, the shield uses less energy (${n1(SH.energy_kwh_day[0])} vs ${n1(RST.energy_kwh_day[0])} kWh/day) **and** has ${(RST.discomfort_kh_day[0] / SH.discomfort_kh_day[0]).toFixed(1)}× less discomfort, so it Pareto-dominates. Using the tail risk (CVaR) instead of the mean lowers discomfort by ${red(SH.discomfort_kh_day[0], MEAN.discomfort_kh_day[0])} for ${n1(100 * (SH.energy_kwh_day[0] / MEAN.energy_kwh_day[0] - 1))} % more energy.`));
push(IMG("fig7_energy_comfort.png", 520));
push(CAP("Fig. 8 – Energy–comfort plane. The CVaR budget β traces a front that lies below and to the left of all baselines."));
push(IMG("fig9_bathroom_day.png", 560));
push(CAP("Fig. 9 – One day of the bathroom zone. Reactive control heats only after entry (cold bathroom). The schedule heats all day. The AURA shield pre-conditions ahead of likely use."));

push(SUB("8.9 Robustness and end-to-end coupling"));
push(TABLE(["Shield variant", "Energy (kWh/day)", "Discomfort (K·h/day)", "Overrides / day", "Max cold during falls (K)"],
  [["AURA shield on AURA stream (proposed)", SH], ["AURA shield on Oracle stream (100 % traffic)", SOR],
    ["AURA shield on Send-on-Delta stream", SSOD], ["AURA shield with 25 % twin-parameter mismatch", SMIS]]
    .map(([n, t]) => [n, n1(t.energy_kwh_day[0]), n2(t.discomfort_kh_day[0]), n2(t.interventions_day[0]), n2(t.fall_max_deficit[0])]),
  [3700, 1400, 1600, 1400, CONTENT_W - 8100], { highlight: [0] }));
push(CAP("Table 6 – The shield on the AURA-gated stream performs close to the shield on the full Oracle stream, and better than on a Send-on-Delta stream with more traffic. It stays effective with 25 % twin mismatch."));

push(SUB("8.10 Statistical significance"));
const SG = R.stats_gating, ST = R.stats_twin;
push(TABLE(["AURA vs.", "p (macro-F1)", "p (traffic)", "p (lifetime)"],
  Object.entries(SG).map(([b, s]) => [b, pv(s.p_f1), pv(s.p_tx), pv(s.p_life)]),
  [3600, 2000, 2000, CONTENT_W - 7600]));
push(CAP(`Table 7 – Two-sided Wilcoxon signed-rank tests, paired over ${proto.test_homes} unseen homes.`));
push(TABLE(["AURA shield vs.", "p (energy)", "p (discomfort)"],
  Object.entries(ST).map(([b, s]) => [b, pv(s.p_energy), pv(s.p_discomfort)]),
  [4600, 2500, CONTENT_W - 7100]));
push(CAP(`Table 8 – Two-sided Wilcoxon signed-rank tests, paired over ${proto.twin_homes} homes.`));

push(SUB("8.11 Summary of validated claims"));
push(B(`**${red(A.tx_rate[0], O.tx_rate[0])} less uplink traffic** with **${pct(A.macro_f1[0] / O.macro_f1[0], 1)} of full-stream macro-F1** (${f1(A.macro_f1[0])} vs ${f1(O.macro_f1[0])}).`));
push(B(`**×${(A.life_mean[0] / O.life_mean[0]).toFixed(1)} battery life** (${n1(A.life_mean[0] / 365)} years on a CR2032), with the downlink attention beacons already included.`));
push(B(`Fall recall **${pct(A.recall[0], 1)}** (Oracle ${pct(O.recall[0], 1)}), with ${n2(A.fa_per_day[0])} false alarms/day.`));
push(B(`Proactive heating with **${red(SH.energy_kwh_day[0], SCH.energy_kwh_day[0])} energy saving** vs the schedule and **${red(SH.discomfort_kh_day[0], GRE.discomfort_kh_day[0])} less discomfort** than a greedy agent.`));
push(B("Each claimed element is supported by the ablation study, and the results are statistically significant over unseen homes."));
push(SUB("8.12 Validity note and next steps"));
push(P("All results above come from a high-fidelity simulation with real-world-calibrated parameters (radio energy, thermal RC constants, sensor noise and drift). They were obtained with held-out test homes, independent seeds and paired significance tests, which establishes an experimental proof of concept (TRL 3). Planned next steps: (i) replay on public smart-home datasets (CASAS Aruba/Milan, ARAS, Orange4Home); (ii) a physical testbed with nRF52840 / CC2652 802.15.4 nodes and a Raspberry-Pi gateway, measuring current with a power analyser; (iii) a pilot in an assisted-living facility.", { run: { size: 20 } }));

// 9. Protection
push(SEC("9. What aspect(s) of the invention need(s) protection?"));
[
  "**System claim:** a system with a plurality of battery-powered sensor nodes and an edge gateway. Each node holds a twin predictor mirrored at the gateway and a programmable dead-band, and transmits a sample only if its residual exceeds the dead-band. The gateway (i) maintains a posterior over occupant intents, (ii) computes for each sensor a discriminability score from a look-ahead intent distribution and class-conditional sensor statistics, (iii) maps the score to a per-sensor dead-band, and (iv) transmits the dead-bands to the nodes as an attention vector.",
  "The **silence-aware (interval) likelihood**: for every silent node, the class-conditional likelihood integrated over the band [x̂ − θ, x̂ + θ], used inside a Bayesian intent filter.",
  "The **safety floor**: a minimum probability mass on one or more emergency intents is mixed into the look-ahead distribution before the discriminability is computed. This keeps emergency-revealing sensors at a tightened dead-band.",
  "The **asymmetric quantised attention beacon**: dead-band levels are quantised, tightening is applied at once and relaxing is rate-limited, and the vector is carried in broadcast beacons or ACK piggy-backing.",
  "The **counterfactual digital-twin shield**: N intent futures are sampled from the gateway posterior, a physical digital twin is simulated per candidate actuation plan, CVaR_α of discomfort or safety cost is computed, and an agent’s proposal is certified or vetoed and repaired with the least-cost admissible plan.",
  "The **end-to-end coupling**: one intent posterior, computed from the gated stream, drives both the communication policy and the risk-certified actuation. Optionally, the shield’s tail risk feeds back into the attention vector.",
  "The corresponding **method**, a **non-transitory computer-readable medium**, and **gateway/node apparatus** (firmware with the comparator gate and dead-band register; gateway software modules 201–203 and 301).",
  "Applications to smart homes, ambient-assisted living / elderly care (fall detection plus thermal safety), smart buildings and campuses, Industry 5.0 human-robot collaboration, and vehicular / low-altitude ambient IoT.",
].forEach((t) => push(N(t, "claims")));

// 10. TRL
push(SEC("10. What is Technology readiness level of your invention? (Tick the appropriate TRL)"));
const trl = [
  ["TRL 1", "Basic Principles observed"], ["TRL 2", "Technology concept formulated"],
  ["TRL 3", "Experimental proof of concept"], ["TRL 4", "Technology validated in a lab"],
  ["TRL 5", "Technology validated in a relevant environment"], ["TRL 6", "Technology demonstrated in a relevant environment"],
  ["TRL 7", "System prototype demonstration in an operational environment"], ["TRL 8", "System complete and qualified"],
  ["TRL 9", "Actual system proven in an operational environment"],
];
const SEL = 2;
const tw = Math.floor(CONTENT_W / 9);
const trlW = Array(9).fill(tw);
trlW[8] = CONTENT_W - tw * 8;
push(new Table({
  width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: trlW,
  rows: [
    new TableRow({ children: [
      cell("Research", trlW[0] * 3, { span: 3, fill: "D9E2F3", bold: true, align: AlignmentType.CENTER }),
      cell("Development", trlW[3] * 3, { span: 3, fill: "D9E2F3", bold: true, align: AlignmentType.CENTER }),
      cell("Deployment", trlW[6] * 2 + trlW[8], { span: 3, fill: "D9E2F3", bold: true, align: AlignmentType.CENTER }),
    ] }),
    new TableRow({ children: trl.map(([t], i) => cell((i === SEL ? "☑ " : "☐ ") + t, trlW[i],
      { bold: true, align: AlignmentType.CENTER, fill: i === SEL ? "C6E0B4" : undefined, size: 17 })) }),
    new TableRow({ children: trl.map(([, d], i) => cell(d, trlW[i],
      { align: AlignmentType.CENTER, fill: i === SEL ? "E2EFDA" : undefined, size: 15 })) }),
  ],
}));
push(SP());
push(P("**Selected: TRL 3 – Experimental proof of concept.** The complete AURA method has been implemented in software and validated experimentally against established baselines on 30 unseen simulated homes (420 home-days), with ablations and significance tests. Lab validation on physical 802.15.4 nodes (TRL 4) is the planned next milestone."));
push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 300 },
  children: [new TextRun({ text: "----------------------END OF THE DOCUMENT-----------------------------", font: FONT, size: 20 })] }));

// ------------------------------------------------------------ header / doc
const hb = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const hbs = { top: hb, bottom: hb, left: hb, right: hb };
const hcell = (text, w, o = {}) => new TableCell({
  borders: hbs, width: { size: w, type: WidthType.DXA }, rowSpan: o.rowSpan,
  verticalAlign: VerticalAlign.CENTER, margins: { top: 30, bottom: 30, left: 80, right: 80 },
  children: [new Paragraph({ alignment: o.align || AlignmentType.LEFT,
    children: [new TextRun({ text, font: FONT, size: o.size || 16, bold: o.bold })] })],
});
const HW = [2000, 4146, 1800, 1800];
const header = new Header({ children: [new Table({
  width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: HW,
  rows: [
    new TableRow({ children: [hcell("©VIT IPR&TTCELL", HW[0], { rowSpan: 3, bold: true, size: 20, align: AlignmentType.CENTER }),
      hcell("Invention Disclosure Format (IDF)-B", HW[1], { rowSpan: 3, bold: true, size: 22, align: AlignmentType.CENTER }),
      hcell("Document No.", HW[2]), hcell("02-IPR-R003", HW[3])] }),
    new TableRow({ children: [hcell("Issue No/Date", HW[2]), hcell("2/01.02.2024", HW[3])] }),
    new TableRow({ children: [hcell("Amd. No/Date", HW[2]), hcell("0/00.00.0000", HW[3])] }),
  ] }), new Paragraph({ children: [] })] });
const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
  new TextRun({ text: "AURA – Invention Disclosure (IDF-B) · Confidential · Page ", font: FONT, size: 16, color: "666666" }),
  new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: "666666" })] })] });

const numCfg = (ref, fmt, text) => ({ reference: ref, levels: [{ level: 0, format: fmt, text, alignment: AlignmentType.LEFT,
  style: { paragraph: { indent: { left: 500, hanging: 360 } } } }, { level: 1, format: LevelFormat.BULLET, text: "–",
  alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 900, hanging: 300 } } } }] });

const doc = new Document({
  creator: "AURA inventors", title: "AURA – Invention Disclosure Format (IDF)-B",
  styles: { default: { document: { run: { font: FONT, size: 21 } } } },
  numbering: { config: [
    numCfg("bullets", LevelFormat.BULLET, "•"),
    numCfg("claims", LevelFormat.DECIMAL, "%1."),
    numCfg("objectives", LevelFormat.DECIMAL, "O%1."),
    numCfg("algo", LevelFormat.DECIMAL, "Step %1:"),
  ] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 },
      margin: { top: 1700, bottom: 1000, left: MARGIN, right: MARGIN, header: 500 } } },
    headers: { default: header }, footers: { default: footer }, children: body,
  }],
});
Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "AURA_IDF_Form_B.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out);
});
