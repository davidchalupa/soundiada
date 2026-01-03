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
};

const EmbeddingViewer: React.FC<Props> = ({ padding = 40 }) => {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playingFile, setPlayingFile] = useState<string | null>(null);
  const [viewport, setViewport] = useState({ w: window.innerWidth, h: window.innerHeight });

  // canonical initial viewport (captured on first mount)
  const canonicalRef = useRef<{ w: number; h: number; portrait: boolean } | null>(null);

  const [guesses, setGuesses] = useState<Record<string, string>>({});
  const [showModal, setShowModal] = useState(false);

  // set canonical viewport on first mount
  useEffect(() => {
    if (!canonicalRef.current) {
      const w0 = Math.max(200, window.innerWidth);
      const h0 = Math.max(160, window.innerHeight);
      const portrait0 = !!(window.matchMedia && window.matchMedia("(orientation: portrait)").matches);
      canonicalRef.current = { w: w0, h: h0, portrait: portrait0 };
    }
  }, []);

  // update viewport on resize/orientation changes
  useEffect(() => {
    const update = () => setViewport({ w: Math.max(200, window.innerWidth), h: Math.max(160, window.innerHeight) });
    update();
    window.addEventListener("resize", update, { passive: true });
    window.addEventListener("orientationchange", update, { passive: true });
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("orientationchange", update);
    };
  }, []);

  const embeddings = useMemo(() => esc50Embeddings.slice(), []);

  const totalFlagged = useMemo(() => Object.keys(guesses).length, [guesses]);
  const correctCount = useMemo(
    () => embeddings.reduce((acc, e) => (guesses[e.file] === e.category ? acc + 1 : acc), 0),
    [guesses, embeddings]
  );

  useEffect(() => {
    if (embeddings.length > 0 && totalFlagged === embeddings.length) {
      setShowModal(true);
      audioRef.current?.pause();
    } else {
      setShowModal(false);
    }
  }, [totalFlagged, embeddings.length]);

  // logical canonical canvas (use initial orientation dims as-is)
  const canonical = canonicalRef.current ?? { w: viewport.w, h: viewport.h, portrait: viewport.h > viewport.w };
  const cw0 = canonical.w;
  const ch0 = canonical.h;
  const initialPortrait = canonical.portrait;

  // detect whether current orientation differs from canonical
  const currentPortrait = viewport.h > viewport.w;
  const rotated = initialPortrait !== currentPortrait;

  // compute scale to fit canonical canvas into current viewport
  // if rotated, the canonical canvas is rotated 90deg when mapping to viewport, so swap dims
  const scale = (() => {
    if (rotated) {
      // canonical (cw0 x ch0) will be mapped rotated -> compare viewport.w to ch0 and viewport.h to cw0
      return Math.min(viewport.w / ch0, viewport.h / cw0);
    } else {
      return Math.min(viewport.w / cw0, viewport.h / ch0);
    }
  })();

  // compute the transform for the canonical content container:
  // center container at viewport center and then rotate + scale as needed.
  // Using translate(-50%,-50%) centers the element (which has width cw0 and height ch0),
  // then rotate(90deg) (if needed) then scale(scale).
  const transform = rotated
    ? `translate(-50%,-50%) rotate(90deg) scale(${scale})`
    : `translate(-50%,-50%) scale(${scale})`;

  // projection bounds based on canonical logical coords (cw0 x ch0)
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

  // project using canonical dims (cw0 x ch0)
  const project = (x: number, y: number) => {
    const { minX, maxX, minY, maxY } = bounds;
    const w = cw0 - 2 * padding;
    const h = ch0 - 2 * padding;
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

  // tiny deterministic jitter
  const jitter = useMemo(() => {
    const xs = embeddings.map((e) => Number(e.x));
    if (xs.length === 0) return 0.001;
    return 1e-3 * Math.max(1.0, Math.max(...xs) - Math.min(...xs));
  }, [embeddings]);

  // audio behavior
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

  // pointer drag fallback + ghost
  const draggingCatRef = useRef<string | null>(null);
  const pointerIdRef = useRef<number | null>(null);
  const ghostRef = useRef<HTMLElement | null>(null);

  const makeGhost = (cat: string, x: number, y: number) => {
    if (ghostRef.current && ghostRef.current.parentNode) ghostRef.current.parentNode.removeChild(ghostRef.current);
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

  const onPointerUpWindow = (ev: PointerEvent) => {
    try {
      const el = document.elementFromPoint(ev.clientX, ev.clientY) as HTMLElement | null;
      if (el) {
        const pt = el.closest(".embed-point") as HTMLElement | null;
        if (pt) {
          const file = pt.getAttribute("data-file");
          if (file && draggingCatRef.current) {
            setGuesses((prev) => ({ ...prev, [file]: draggingCatRef.current as string }));
          }
        }
      }
    } finally {
      cleanupGhost();
      draggingCatRef.current = null;
      pointerIdRef.current = null;
      window.removeEventListener("pointermove", onPointerMoveWindow);
      window.removeEventListener("pointerup", onPointerUpWindow);
    }
  };

  const startPointerDrag = (ev: React.PointerEvent, category: string) => {
    try {
      ev.preventDefault();
      ev.stopPropagation();
    } catch {}
    try {
      (ev.currentTarget as Element).setPointerCapture?.(ev.pointerId);
    } catch {}
    draggingCatRef.current = category;
    pointerIdRef.current = ev.pointerId;
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

  // desktop drag
  const onPaletteDragStart = (ev: React.DragEvent, category: string) => {
    try {
      ev.dataTransfer.setData("text/plain", category);
    } catch {}
  };

  const onPointDragOver = (ev: React.DragEvent) => ev.preventDefault();
  const onPointDrop = (ev: React.DragEvent, file: string) => {
    ev.preventDefault();
    const cat = ev.dataTransfer.getData("text/plain");
    if (cat && typeof cat === "string") setGuesses((prev) => ({ ...prev, [file]: cat }));
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

  // container & content styles:
  // wrapper fills viewport; content (canonical canvas) is centered and transformed
  const wrapperStyle: React.CSSProperties = {
    position: "fixed",
    left: 0,
    top: 0,
    width: "100vw",
    height: "100vh",
    overflow: "hidden",
    zIndex: 1,
  };

  const contentStyle: React.CSSProperties = {
    position: "absolute",
    left: "50%",
    top: "50%",
    width: `${cw0}px`,
    height: `${ch0}px`,
    transformOrigin: "center center",
    transform,
    pointerEvents: "auto",
    touchAction: "none",
    // ensure content sits below palette/instruction
    zIndex: 1,
  };

  return (
    <div style={{ position: "relative", width: "100vw", height: "100vh", overflow: "hidden" }}>
      {/* instructions: fixed top-left */}
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
        <div>Tap a circle to play the sound.</div>
        <div>Drag an icon from the palette (bottom-right) onto a circle to label it.</div>
      </div>

      {/* main wrapper */}
      <div style={wrapperStyle}>
        <div style={contentStyle}>
          <svg width={cw0} height={ch0} viewBox={`0 0 ${cw0} ${ch0}`} preserveAspectRatio="none">
            <rect x={0} y={0} width={cw0} height={ch0} fill="#071030" />
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
        </div>
      </div>

      {/* fixed bottom-right palette */}
      <div
        className="palette-root"
        aria-hidden
        style={{
          position: "fixed",
          right: 18,
          bottom: 18,
          display: "flex",
          gap: 10,
          zIndex: 1700,
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
    let g = ((num >> 8) & 0x00ff) + amt;
    let b = (num & 0x0000ff) + amt;
    r = Math.max(0, Math.min(255, r));
    g = Math.max(0, Math.min(255, g));
    b = Math.max(0, Math.min(255, b));
    return `rgb(${r},${g},${b})`;
  } catch {
    return hex;
  }
}
