// T306: checks the built kit against firebase.json and the instances' .env
// files, so CI catches what would otherwise stall or misplace a deploy:
// - every instance sets exactly the settings the kit declares (a kit update
//   that adds one would make the deploy stop and ask);
// - every function runs in the region index.ts sets.
// Run after `npm run build`: `npm run check`.
const fs = require("node:fs");
const path = require("node:path");
const { declaredParams } = require("firebase-functions/params");
const functions = require("./lib/index.js");

const REGION = "asia-southeast1";
const root = path.resolve(__dirname, "..");
const config = JSON.parse(fs.readFileSync(path.join(root, "firebase.json"), "utf8"));
const kit = config.functions.find((entry) => entry.kit);
const declared = declaredParams.map((param) => param.name).sort();
const problems = [];

for (const [instance, dir] of Object.entries(kit.instances)) {
  const lines = fs.readFileSync(path.join(root, dir, ".env"), "utf8").split("\n");
  const keys = lines
    .filter((line) => /^[A-Z_]+=/.test(line))
    .map((line) => line.split("=")[0])
    .sort();
  const missing = declared.filter((key) => !keys.includes(key));
  const unknown = keys.filter((key) => !declared.includes(key));
  if (missing.length > 0) problems.push(`${instance}: missing ${missing.join(", ")}`);
  if (unknown.length > 0) problems.push(`${instance}: not a kit setting: ${unknown.join(", ")}`);
}

// The deployed functions are the exports with an endpoint; the rest are helpers.
const deployed = Object.entries(functions).filter(([, fn]) => fn?.__endpoint);
for (const [name, fn] of deployed) {
  const regions = fn.__endpoint.region;
  if (JSON.stringify(regions) !== JSON.stringify([REGION])) {
    problems.push(`${name} runs in ${regions}, not ${REGION}`);
  }
}

if (problems.length > 0) {
  console.error(problems.join("\n"));
  process.exitCode = 1;
} else {
  const instances = Object.keys(kit.instances).length;
  console.log(`${instances} instances set all ${declared.length} kit settings;`);
  console.log(`${deployed.length} functions per instance run in ${REGION}.`);
}
