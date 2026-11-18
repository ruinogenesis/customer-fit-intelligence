// The bare love matrix (T206): every customer as a dot, live from Firestore.
//
// For now it reads the local emulator only. The emulator accepts an unsigned
// stand-in token, so the page reads as a "viewer" without a sign-in screen;
// staff sign-in comes with the dashboard (T401).

import { initializeApp } from "https://www.gstatic.com/firebasejs/13.3.0/firebase-app.js";
import {
  collection,
  connectFirestoreEmulator,
  getFirestore,
  onSnapshot,
} from "https://www.gstatic.com/firebasejs/13.3.0/firebase-firestore.js";
import {
  LINE,
  MARGIN,
  QUADRANTS,
  SIZE,
  along,
  countByQuadrant,
  toDot,
} from "./matrix.js";

const SVG = "http://www.w3.org/2000/svg";
// The Firestore emulator, as firebase.json and emulators:start set it up.
const EMULATOR_HOST = "127.0.0.1";
const EMULATOR_PORT = 8080;
const LOCAL_HOSTS = ["localhost", "127.0.0.1"];

const status = document.querySelector("#status");
const matrix = document.querySelector("#matrix");
const frame = document.querySelector("#frame");
const layer = document.querySelector("#dots");
const counts = document.querySelector("#counts tbody");

function svg(name, attributes = {}, text = "") {
  const element = document.createElementNS(SVG, name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }
  element.textContent = text;
  return element;
}

// The plot's border, D5's two lines, the quadrant names and the axis labels.
function drawFrame() {
  const low = along(0);
  const high = along(1);
  const middle = along(LINE);
  const side = high - low;
  const loveAxis = `translate(${MARGIN / 3} ${SIZE / 2}) rotate(-90)`;
  matrix.setAttribute("viewBox", `0 0 ${SIZE} ${SIZE}`);
  frame.append(
    svg("rect", { class: "plot", x: low, y: low, width: side, height: side }),
    svg("line", { class: "line", x1: middle, y1: low, x2: middle, y2: high }),
    svg("line", { class: "line", x1: low, y1: middle, x2: high, y2: middle }),
    svg("text", { class: "quadrant", x: low + 8, y: low + 20 }, QUADRANTS.fad),
    svg("text", { class: "quadrant", x: middle + 8, y: low + 20 }, QUADRANTS.lovemark),
    svg("text", { class: "quadrant", x: low + 8, y: middle + 20 }, QUADRANTS.product),
    svg("text", { class: "quadrant", x: middle + 8, y: middle + 20 }, QUADRANTS.brand),
    svg("text", { class: "axis", x: SIZE / 2, y: SIZE - MARGIN / 3 }, "Respect →"),
    svg("text", { class: "axis", x: 0, y: 0, transform: loveAxis }, "Love →"),
  );
}

// One <circle> per customer, kept across snapshots so that a move glides.
const circles = new Map();

function show(dots) {
  const seen = new Set();
  for (const dot of dots) {
    seen.add(dot.id);
    let circle = circles.get(dot.id);
    if (circle === undefined) {
      circle = svg("circle", { r: 5 });
      circle.append(svg("title"));
      circles.set(dot.id, circle);
      layer.append(circle);
    }
    circle.setAttribute("class", `dot ${dot.quadrant}`);
    circle.style.transform = `translate(${dot.x}px, ${dot.y}px)`;
    circle.firstChild.textContent = `${dot.id}: ${QUADRANTS[dot.quadrant]}`;
  }
  for (const [id, circle] of circles) {
    if (!seen.has(id)) {
      circle.remove();
      circles.delete(id);
    }
  }
  counts.replaceChildren(
    ...Object.entries(countByQuadrant(dots)).map(([quadrant, count]) => {
      const row = document.createElement("tr");
      const name = document.createElement("th");
      const value = document.createElement("td");
      name.scope = "row";
      name.textContent = QUADRANTS[quadrant];
      value.textContent = String(count);
      row.append(name, value);
      return row;
    }),
  );
}

function start() {
  drawFrame();
  if (!LOCAL_HOSTS.includes(location.hostname)) {
    status.textContent =
      "For now this page reads the local Firestore emulator only: " +
      "open it from firebase emulators:start.";
    return;
  }
  const app = initializeApp({ projectId: "demo-cfi", apiKey: "demo-key" });
  const db = getFirestore(app);
  connectFirestoreEmulator(db, EMULATOR_HOST, EMULATOR_PORT, {
    mockUserToken: { sub: "local-viewer", role: "viewer" },
  });
  onSnapshot(
    collection(db, "customers"),
    (snapshot) => {
      const dots = snapshot.docs
        .map((doc) => toDot(doc.id, doc.data()))
        .filter((dot) => dot !== null);
      show(dots);
      const time = new Date().toLocaleTimeString();
      status.textContent = `${dots.length} customers, live (updated ${time}).`;
    },
    (error) => {
      status.textContent = `Firestore refused to share customers: ${error.message}`;
    },
  );
}

start();
