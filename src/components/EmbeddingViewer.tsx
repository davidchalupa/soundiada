// src/components/EmbeddingViewer.tsx
import React, { useRef, useState, useMemo, useEffect } from "react";
import { esc50Embeddings } from "../resources/embeddings";
import "./EmbeddingViewer.css";

import IconSiren from "../assets/icons/siren.png";
import IconBird from "../assets/icons/bird.png";
import IconBreathing from "../assets/icons/breath.png";
import IconWashingMachine from "../assets/icons/washing-machine.png";
import IconCryingBaby from "../assets/icons/crying.png";

const PALETTE_CATEGORIES = [
  "siren",
  "chirping_birds",
  "breathing",
  "washing_machine",
  "crying_baby",
] as const;

const CATEGORY_COLOR: Record<string, string> = {
  siren: "#ff6b6b",
  chirping_birds: "#ffd166",
  breathing: "#6be4b0",
  washing_machine: "#74c0ff",
  crying_baby: "#d291ff",
};

function buildAudioUrl(filename: string) {
  return `https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/${filename}`;
}

type Props = {
  padding?: number;
  aspectRatio?: number; // height/width, default 9/16
};

const EmbeddingViewer: React.FC<Props> = ({ padding = 40, aspectRatio = 9 / 16 }) => {
  // refs & state
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playingFile, setPlayingFile] = useState<string | null>(null);
  const [viewport, setViewport] = useState({ w: window.innerWidth, h: window.innerHeight });
  const [guesses, setGuesses] = useState<Record<string, string>>({});
  const [showModal, setShowModal] = useState(false);

  // Drag active tracking (used to avoid showing modal while user drags)
  const isDraggingRef = useRef(false);
  const [isDragging, setIsDragging] = useState(false);

  // update viewport
  useEffect(() => {
    const update = () =>
      setViewport({
        w: Math.max(200, window.innerWidth),
        h: Math.max(160, window.innerHeight),
      });
    update();
    window.addEventListener("resize", update, { passive: true });
    window.addEventListener("orientationchange", update, { passive: true });
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("orientationchange", update);
    };
  }, []);

  const isLandscape = viewport.w >= viewport.h;

  // embeddings frozen copy
  const embeddings = useMemo(() => esc50Embeddings.slice(), []);

  const totalFlagged = useMemo(() => {
    // count only files that are in the current embedding list and have a valid guess
    const validSet = new Set(embeddings.map((e) => e.file));
    let cnt = 0;
    for (const f of embeddings.map((e) => e.file)) {
      if (guesses[f] && typeof guesses[f] === "string") cnt++;
    }
    return cnt;
  }, [guesses, embeddings]);

  const correctCount = useMemo(
    () => embeddings.reduce((acc, e) => (guesses[e.file] === e.category ? acc + 1 : acc), 0),
    [guesses, embeddings]
  );

  // only show modal when all embedding files are labeled AND no drag is active
  useEffect(() => {
    // compute labeled count precisely from visible embeddings
    const labeledCount = embeddings.reduce((acc, e) => (guesses[e.file] ? acc + 1 : acc), 0);

    // debounce slightly to avoid race with pointerup/dragend
    let t: number | null = null;
    if (labeledCount === embeddings.length && !isDraggingRef.current) {
      t = window.setTimeout(() => {
        // double-check guard
        if (embeddings.reduce((a, e) => (guesses[e.file] ? a + 1 : a), 0) === embeddings.length && !isDraggingRef.current) {
          audioRef.current?.pause();
          setShowModal(true);
        }
      }, 120);
    } else {
      setShowModal(false);
    }
    return () => {
      if (t) window.clearTimeout(t);
    };
  }, [guesses, embeddings, isDragging]);

  // canvas sizing (width fills viewport; height capped to aspect ratio)
  const canvasW = viewport.w;
  const canvasH = Math.min(viewport.h, Math.round(viewport.w * aspectRatio));
  const canvasTop = Math.round((viewport.h - canvasH) / 2); // vertical centering when shorter

  // bounds for projection
  const bounds = useMemo(() => {
    if (!embeddings || embeddings.length === 0) return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (const e of embeddings) {
      const x = Number(e.x),
        y = Number(e.y);
      if (Number.isFinite(x) && Number.isFinite(y)) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
    }
    if (minX === maxX) {
      minX -= 1;
      maxX += 1;
    }
    if (minY === maxY) {
      minY -= 1;
      maxY += 1;
    }
    return { minX, maxX, minY, maxY };
  }, [embeddings]);

  // project into canvas coordinates
  const project = (x: number, y: number) => {
    const { minX, maxX, minY, maxY } = bounds;
    const w = canvasW - 2 * padding;
    const h = canvasH - 2 * padding;
    const px = padding + ((x - minX) / (maxX - minX)) * w;
    const py = padding + (1 - (y - minY) / (maxY - minY)) * h;
    return { px, py };
  };

  function getCategoryIcon(cat: string) {
    switch (cat) {
      case "siren":
        return IconSiren;
      case "chirping_birds":
        return IconBird;
      case "breathing":
        return IconBreathing;
      case "washing_machine":
        return IconWashingMachine;
      case "crying_baby":
        return IconCryingBaby;
      default:
        return IconSiren;
    }
  }

  // jitter to avoid complete overplot
  const jitter = useMemo(() => {
    const xs = embeddings.map((e) => Number(e.x));
    if (xs.length === 0) return 0.001;
    return 1e-3 * Math.max(1.0, Math.max(...xs) - Math.min(...xs));
  }, [embeddings]);

  // play/pause
  const onPointClick = (file: string) => {
    const url = buildAudioUrl(file);
    const audio = audioRef.current;
    if (!audio) return;
    const currentSrcHasFile = audio.src && audio.src.includes(file);
    if (currentSrcHasFile && !audio.paused) {
      audio.pause();
      setPlayingFile(null);
      return;
    }
    if (currentSrcHasFile && audio.paused) {
      audio.play().then(() => setPlayingFile(file)).catch(() => setPlayingFile(null));
      return;
    }
    audio.pause();
    audio.src = url;
    audio.currentTime = 0;
    const p = audio.play();
    if (p !== undefined) {
      p.then(() => setPlayingFile(file)).catch(() => setPlayingFile(null));
    } else {
      setPlayingFile(file);
    }
  };

  useEffect(() => {
    const a = audioRef.current;
    if (!a) return;
    const onEnded = () => setPlayingFile(null);
    const onPause = () => {
      if (a.currentTime > 0 && !a.ended) setPlayingFile(null);
    };
    a.addEventListener("ended", onEnded);
    a.addEventListener("pause", onPause);
    return () => {
      a.removeEventListener("ended", onEnded);
      a.removeEventListener("pause", onPause);
    };
  }, []);

  // Pointer-based drag (touch/mouse) with ghost element
  const draggingCatRef = useRef<string | null>(null);
  const ghostRef = useRef<HTMLElement | null>(null);

  const makeGhost = (cat: string, x: number, y: number) => {
    cleanupGhost();
    const g = document.createElement("div");
    g.style.position = "fixed";
    g.style.left = `${x - 20}px`;
    g.style.top = `${y - 20}px`;
    g.style.width = "40px";
    g.style.height = "40px";
    g.style.zIndex = "9999";
    g.style.pointerEvents = "none";
    g.style.borderRadius = "6px";
    g.style.backgroundImage = `url(${getCategoryIcon(cat)})`;
    g.style.backgroundSize = "cover";
    g.style.boxShadow = "0 6px 18px rgba(0,0,0,0.35)";
    document.body.appendChild(g);
    ghostRef.current = g;
  };

  const cleanupGhost = () => {
    if (ghostRef.current && ghostRef.current.parentNode) {
      ghostRef.current.parentNode.removeChild(ghostRef.current);
      ghostRef.current = null;
    }
  };

  const onPointerMoveWindow = (ev: PointerEvent) => {
    if (!ghostRef.current) return;
    ghostRef.current.style.left = `${ev.clientX - 20}px`;
    ghostRef.current.style.top = `${ev.clientY - 20}px`;
  };

  // --- IMPORTANT FIX: use geometry-based nearest-point pick on pointer up ---
  const onPointerUpWindow = (ev: PointerEvent) => {
    try {
      // compute canvas-local coordinates:
      const canvasX = ev.clientX;
      const canvasY = ev.clientY - canvasTop;

      // larger adaptive radius for forgiving drops:
      const radius = Math.max(48, Math.round(Math.min(canvasW, canvasH) * 0.075)); // min 48px or 7.5% of min dimension
      const radius2 = radius * radius;

      // find nearest plotted point in pixel coordinates
      let bestIdx = -1;
      let bestDist2 = Infinity;
      for (let i = 0; i < embeddings.length; i++) {
        const e = embeddings[i];
        const x = Number(e.x) + ((i % 2 === 0 ? 1 : -1) * jitter * 0.5);
        const y = Number(e.y) + ((i % 3 === 0 ? 1 : -1) * jitter * 0.5);
        const { px, py } = project(x, y);
        const dx = px - canvasX;
        const dy = py - canvasY;
        const d2 = dx * dx + dy * dy;
        if (d2 < bestDist2) {
          bestDist2 = d2;
          bestIdx = i;
        }
      }

      if (bestIdx >= 0 && bestDist2 <= radius2 && draggingCatRef.current) {
        const file = embeddings[bestIdx].file;
        setGuesses((prev) => ({ ...prev, [file]: draggingCatRef.current as string }));
      } else {
        // fallback: attempt DOM hit-test (rare), but keep geometry primary
        const el = document.elementFromPoint(ev.clientX, ev.clientY) as HTMLElement | null;
        if (el) {
          const pt = el.closest(".embed-point") as HTMLElement | null;
          if (pt) {
            const fileAttr = pt.getAttribute("data-file");
            const cat = draggingCatRef.current;
            if (fileAttr && cat) setGuesses((prev) => ({ ...prev, [fileAttr]: cat }));
          }
        }
      }
    } finally {
      cleanupGhost();
      draggingCatRef.current = null;
      isDraggingRef.current = false;
      setIsDragging(false);
      window.removeEventListener("pointermove", onPointerMoveWindow);
      window.removeEventListener("pointerup", onPointerUpWindow);
    }
  };

  const startPointerDrag = (ev: React.PointerEvent, category: string) => {
    try {
      ev.preventDefault();
      ev.stopPropagation();
    } catch {}
    draggingCatRef.current = category;
    isDraggingRef.current = true;
    setIsDragging(true);
    makeGhost(category, ev.clientX, ev.clientY);
    window.addEventListener("pointermove", onPointerMoveWindow, { passive: true } as any);
    window.addEventListener("pointerup", onPointerUpWindow);
  };

  useEffect(() => {
    return () => {
      cleanupGhost();
      window.removeEventListener("pointermove", onPointerMoveWindow);
      window.removeEventListener("pointerup", onPointerUpWindow);
    };
  }, []);

  // Desktop drag/drop handlers (kept) - also mark dragging state
  const onPaletteDragStart = (ev: React.DragEvent, category: string) => {
    try {
      ev.dataTransfer.setData("text/plain", category);
      isDraggingRef.current = true;
      setIsDragging(true);
    } catch {}
  };

  // ensure we clear dragging state on native dragend/drop anywhere
  useEffect(() => {
    const handleDragEnd = () => {
      isDraggingRef.current = false;
      setIsDragging(false);
    };
    const handleWindowDrop = () => {
      isDraggingRef.current = false;
      setIsDragging(false);
    };
    window.addEventListener("dragend", handleDragEnd);
    window.addEventListener("drop", handleWindowDrop);
    return () => {
      window.removeEventListener("dragend", handleDragEnd);
      window.removeEventListener("drop", handleWindowDrop);
    };
  }, []);

  const onPointDragOver = (ev: React.DragEvent) => ev.preventDefault();
  const onPointDrop = (ev: React.DragEvent, file: string) => {
    ev.preventDefault();
    const cat = ev.dataTransfer.getData("text/plain");
    if (cat && typeof cat === "string") setGuesses((prev) => ({ ...prev, [file]: cat }));
    // clear native dragging state
    isDraggingRef.current = false;
    setIsDragging(false);
  };

  const onPointDoubleClick = (file: string) => {
    setGuesses((prev) => {
      const copy = { ...prev };
      delete copy[file];
      return copy;
    });
  };

  const onCloseModal = () => setShowModal(false);
  const onResetGuesses = () => {
    setGuesses({});
    setShowModal(false);
  };

  // Render
  return (
    <div
      style={{
        position: "fixed",
        left: 0,
        top: 0,
        width: "100vw",
        height: "100vh",
        overflow: "hidden",
        boxSizing: "border-box",
        margin: 0,
        padding: 0,
      }}
    >
      {/* Portrait overlay */}
      {!isLandscape && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "#071030",
            color: "#fff",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 3000,
            padding: 20,
            textAlign: "center",
          }}
        >
          <div>
            <h2 style={{ marginTop: 0 }}>Please rotate your device</h2>
            <p style={{ opacity: 0.9, marginBottom: 12 }}>
              This game is optimized for <strong>landscape</strong> orientation. Turn your phone sideways to continue.
            </p>
            <div style={{ opacity: 0.8, fontSize: 13 }}>When in landscape the app will display correctly.</div>
          </div>
        </div>
      )}

      {/* Top-left instructions */}
      <div
        style={{
          position: "fixed",
          top: 12,
          left: 12,
          zIndex: 1600,
          background: "rgba(255,255,255,0.04)",
          color: "#fff",
          padding: "8px 12px",
          borderRadius: 8,
          fontSize: 13,
          lineHeight: 1.2,
          boxShadow: "0 6px 18px rgba(0,0,0,0.45)",
          pointerEvents: "none",
          userSelect: "none",
        }}
        aria-hidden
      >
        <strong style={{ display: "block", marginBottom: 4 }}>How to play</strong>
        <div>Click a circle to play the sound.</div>
        <div>Drag an icon from the palette (bottom-right) onto a circle to label it.</div>
      </div>

      {/* Canvas */}
      <svg
        width={canvasW}
        height={canvasH}
        viewBox={`0 0 ${canvasW} ${canvasH}`}
        preserveAspectRatio="none"
        style={{
          display: isLandscape ? "block" : "none",
          width: "100vw",
          height: `${canvasH}px`,
          position: "fixed",
          left: 0,
          top: `${canvasTop}px`,
          zIndex: 1,
          boxSizing: "border-box",
          margin: 0,
          padding: 0,
        }}
      >
        <rect x={0} y={0} width={canvasW} height={canvasH} fill="#071030" />
        {embeddings.map((e, idx) => {
          const x = Number(e.x) + ((idx % 2 === 0 ? 1 : -1) * jitter * 0.5);
          const y = Number(e.y) + ((idx % 3 === 0 ? 1 : -1) * jitter * 0.5);
          const { px, py } = project(x, y);
          const isPlaying = playingFile === e.file;
          const guessed = guesses[e.file];
          const fillColor = guessed ? (CATEGORY_COLOR[guessed] ?? "#e0e0e0") : isPlaying ? "#0b5c99" : "#ffffff";
          const strokeColor = guessed ? darken(CATEGORY_COLOR[guessed] ?? "#888") : isPlaying ? "#0e4a7c" : "#6b7280";
          const key = `${e.file}-${idx}`;
          const tooltip = `${e.file} — actual: ${e.category} — guessed: ${guessed ?? "none"}`;

          return (
            <g
              key={key}
              transform={`translate(${px}, ${py})`}
              className="embed-point"
              data-file={e.file}
              style={{ cursor: "pointer" }}
              onClick={() => onPointClick(e.file)}
              onDoubleClick={() => onPointDoubleClick(e.file)}
              onDragOver={onPointDragOver}
              onDrop={(ev) => onPointDrop(ev, e.file)}
              role="button"
              aria-label={tooltip}
            >
              {showModal ? <title>{tooltip}</title> : null}

              <circle r={16} fill="rgba(255,255,255,0.02)" stroke="none" />
              <circle r={12} fill={fillColor} stroke={strokeColor} strokeWidth={isPlaying ? 2.2 : 1.0} />

              {guessed ? (
                <image href={getCategoryIcon(guessed)} x={-12} y={-12} width={24} height={24} style={{ pointerEvents: "none" }} />
              ) : isPlaying ? (
                <>
                  <rect x={-5} y={-7} width={3} height={14} fill="#fff" rx={0.6} />
                  <rect x={1.5} y={-7} width={3} height={14} fill="#fff" rx={0.6} />
                </>
              ) : (
                <polygon points="-4,-6 -4,6 6,0" fill="#111" />
              )}
            </g>
          );
        })}
      </svg>

      {/* palette */}
      <div
        className="palette-root"
        aria-hidden
        style={{
          position: "fixed",
          right: 18,
          bottom: 18,
          display: "flex",
          gap: 10,
          zIndex: 1600,
          touchAction: "none",
        }}
      >
        {PALETTE_CATEGORIES.map((cat) => (
          <div
            key={cat}
            className="palette-item"
            draggable
            onDragStart={(ev) => onPaletteDragStart(ev, cat)}
            onPointerDown={(ev) => startPointerDrag(ev, cat)}
            title={cat.replace(/_/g, " ")}
          >
            <div className="palette-icon">
              <img src={getCategoryIcon(cat)} alt={cat} width={32} height={32} />
            </div>
          </div>
        ))}
      </div>

      {/* modal */}
      {showModal && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(0,0,0,0.55)",
            zIndex: 2000,
          }}
          role="dialog"
          aria-modal="true"
        >
          <div
            style={{
              background: "#071030",
              color: "#fff",
              padding: 24,
              borderRadius: 10,
              minWidth: 320,
              maxWidth: "90%",
              textAlign: "center",
              boxShadow: "0 10px 40px rgba(0,0,0,0.6)",
            }}
          >
            <h2 style={{ margin: 0, marginBottom: 12 }}>Congratulations!</h2>
            <p style={{ margin: 0, marginBottom: 18, fontSize: 16 }}>
              Your final score:{" "}
              <strong style={{ fontSize: 18 }}>
                {correctCount} / {embeddings.length}
              </strong>
            </p>
            <div style={{ display: "flex", gap: 12, justifyContent: "center", marginTop: 8 }}>
              <button
                onClick={onCloseModal}
                style={{
                  padding: "8px 14px",
                  borderRadius: 6,
                  border: "1px solid rgba(255,255,255,0.12)",
                  background: "transparent",
                  color: "#fff",
                  cursor: "pointer",
                }}
              >
                Close
              </button>
              <button
                onClick={onResetGuesses}
                style={{
                  padding: "8px 14px",
                  borderRadius: 6,
                  border: "none",
                  background: "#ff6b6b",
                  color: "#fff",
                  cursor: "pointer",
                }}
              >
                Reset guesses
              </button>
            </div>
          </div>
        </div>
      )}

      <audio ref={audioRef} style={{ display: "none" }} preload="none" />
    </div>
  );
};

export default EmbeddingViewer;

/* helper */
function darken(hex: string, amt = -24) {
  try {
    const parsed = hex.replace("#", "");
    const num = parseInt(parsed, 16);
    let r = (num >> 16) + amt;
    let g = ((num >> 8) & 0x0000ff) + amt;
    let b = (num & 0x0000ff) + amt;
    r = Math.max(0, Math.min(255, r));
    g = Math.max(0, Math.min(255, g));
    b = Math.max(0, Math.min(255, b));
    return `rgb(${r},${g},${b})`;
  } catch {
    return hex;
  }
}
