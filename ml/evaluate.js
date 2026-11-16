// Run the same in-browser ML pipeline under Node to check metrics:  node ml/evaluate.js
const fs = require("fs");
const path = require("path");
const KP = require("../static/ml.js");
const study = JSON.parse(fs.readFileSync(path.join(__dirname, "../static/data/study.json"), "utf8"));

(async () => {
  for (const key of Object.keys(KP.MODEL_SPECS)) {
    const t0 = Date.now();
    const r = await KP.crossValidate(study, key);
    const f = (v) => (v == null ? "  -  " : v.toFixed(3));
    console.log(`${key.padEnd(11)} kss_r ${f(r.kss_r)} kss_mae ${f(r.kss_mae)} | tlx_r ${f(r.tlx_r)} tlx_mae ${f(r.tlx_mae)} | readiness_r ${f(r.rd_r)} mae ${f(r.rd_mae)}  (${Date.now() - t0} ms)`);
  }
  // Shapley sanity: contributions sum to prediction - baseline prediction
  const pipe = KP.trainPipeline(study, "mlp", study.rows);
  const x = pipe.toInput(study.rows[10]);
  const e = KP.explain(pipe, x);
  console.log("shapley additivity error:", Math.abs(e.phi.reduce((a, b) => a + b, 0) - (e.full - e.base)).toExponential(2));
  const imp = KP.globalImportance(pipe, study.rows);
  console.log("importance (kss | tlx):");
  study.features.forEach((k, i) => console.log("  " + k.padEnd(17), imp.kss[i].toFixed(2), "|", imp.tlx[i].toFixed(2)));
})();
