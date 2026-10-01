# AURA — Adaptive Uncertainty-Regulated Ambient intelligence

Reference implementation and experimental validation for the invention disclosure
**"System and method for intent-uncertainty-coupled semantic event gating with
silence-aware inference and counterfactual digital-twin shielding for
energy-efficient, proactive ambient intelligence in AI-native IoT"**.
The topic is aligned with the IEEE IoT Journal special issue *Ambient Intelligence for
AI-Native IoT*.

## The idea in one paragraph

Battery sensor nodes in a smart home keep a tiny *twin predictor* that is mirrored
bit-exactly at the gateway, and transmit only when the measured value leaves a
dead-band θᵢ(t) around the prediction. Unlike send-on-delta or dual-prediction
schemes, **the dead-band is not fixed: the gateway sets it.** The gateway's intent
engine works out which sensors would help tell apart the occupant intents that are
plausible over the next few minutes, and turns that into a per-sensor
*discriminability* Dᵢ(t). It then broadcasts a quantised *attention vector* that
tightens the dead-bands of the sensors that matter right now and relaxes all the
others. A safety floor keeps the sensors that reveal emergencies (such as falls)
alert. Silence also carries information: when a node stays quiet, the gateway knows
the true value lies inside `[x̂ ± θ]` and uses this *interval likelihood*. Finally,
any action an AI agent proposes (here, zone heating) goes through a **counterfactual
digital-twin shield**. The shield rolls a thermal twin forward under many intent
futures sampled from the posterior, and vetoes or repairs plans whose CVaR of
occupant discomfort is above a budget.

## Layout

```
aura/
  config.py       activities, sensors, noise, radio/energy model
  simulator.py    multimodal AAL home simulator (ADLs, falls, artefacts)
  intent.py       time-of-day HMM, silence-aware likelihood, discriminability
  gating.py       Oracle / Periodic / Send-on-Delta / Dual-Prediction / AURA loop
  metrics.py      macro-F1, fall recall/latency/false alarms
  twin.py         4-zone thermal twin, controllers, CVaR shield
experiments/
  run_all.py      full campaign -> results/*.json
  make_figures.py figures/*.png
idf/
  build_idf.js    builds the filled IDF-B form (docx) from the results
tests/            pytest suite
```

## Reproduce

```bash
pip install numpy scipy scikit-learn matplotlib pytest
python -m pytest -q tests
python experiments/run_all.py        # about 1 h on 4 cores
python experiments/make_figures.py
cd idf && node build_idf.js          # -> idf/AURA_IDF_Form_B.docx
```

All results are produced by a simulation: synthetic homes, an RC thermal plant and a
parametric radio energy model (TRL 3, experimental proof of concept). The
hyper-parameters were tuned on validation homes (seeds 1000–1002) that are kept
apart from the 30 test homes (seeds 2000+).
