import "@fontsource-variable/anuphan";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import { firebaseAuth } from "./firebase";
import { reloadOnStaleBuild } from "./staleBuild";
import "./styles.css";

reloadOnStaleBuild();

const root = createRoot(document.getElementById("root")!);
try {
  const auth = await firebaseAuth();
  root.render(
    <StrictMode>
      <App auth={auth} />
    </StrictMode>,
  );
} catch (error) {
  root.render(<p className="startup">The dashboard couldn't start: {String(error)}</p>);
}
