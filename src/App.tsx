// src/App.tsx
import React from "react";
import EmbeddingViewer from "./components/EmbeddingViewer";
import "./index.css";

const App: React.FC = () => {
  return (
    <div style={{ padding: 20, display: "flex", justifyContent: "center" }}>
      <EmbeddingViewer />
    </div>
  );
};

export default App;
