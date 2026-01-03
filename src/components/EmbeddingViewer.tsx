// src/components/EmbeddingViewer.tsx
import React, { useRef, useState, useMemo, useEffect } from "react";
import { esc50Embeddings } from "../resources/embeddings";
import "./EmbeddingViewer.css";

import IconSiren from "../assets/icons/siren.png";
import IconBird from "../assets/icons/bird.png";
import IconBreathing from "../assets/icons/breath.png";
import IconWashingMachine from "../assets/icons/washing-machine.png";
import IconCryingBaby from "../assets/icons/crying.png";

/**
 * - responsive fullscreen canvas
 * - click to play / pause audio
 * - drag & drop labels from palette (desktop HTML5 drag + pointer / touch fallback)
 * - keeps guesses state, shows modal when all items are labeled
 * - portrait-orientation overlay that recommends landscape and tries to lock orientation
 */

/* palette categories (sample) */
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
  // refs
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // UI state
  const [playingFile, setPlayingFile] = useState<string | null>(null);
  const [containerSize, setContainerSize] = useState({ w: window.innerWidth, h: window.innerHeight });

  // guesses: mapping file -> guessed category
  const [guesses, setGuesses] = useState<Record<string, string>>({});

  // modal shown when all samples guessed
  const [showModal, setShowModal] = useState(false);

  // orientation overlay state
  const [isPortrait, setIsPortrait] = useState(() => {
    if (typeof window === "undefined") return false;
    return !!(window.matchMedia && window.matchMedia("(orientation: portrait)").matches);
  });

  // embeddings (frozen copy)
  const embeddings = useMemo(() => esc50Embeddings.slice(), []);

  // counts derived
  const totalFlagged = useMemo(() => Object.keys(guesses).length, [guesses]);
  const correctCount = useMemo(() => embeddings.reduce((acc, e) => (guesses[e.file] === e.category ? acc + 1 : acc), 0), [guesses, embeddings]);

  // show modal when all flagged
  useEffect(() => {
    if (embeddings.length > 0 && totalFlagged === embeddings.length) {
      setShowModal(true);
      audioRef.current?.pause();
    } else {
      setShowModal(false);
    }
  }, [totalFlagged, embeddings.length]);

  // responsive size
  useEffect(() => {
    const update = () => {
      setContainerSize({ w: Math.max(200, window.innerWidth), h: Math.max(160, window.innerHeight) });
    };
    update();
    window.addEventListener("resize", update, { passive: true });
    return () => window.removeEventListener("resize", update);
  }, []);

  // orientation: keep media query + resize sync
  useEffect(() => {
    if (typeof window === "undefined") return;
    const mq = window.matchMedia("(orientation: portrait)");
    const mqHandler = (ev: any) => setIsPortrait(!!ev.matches);
    if (typeof mq.addEventListener === "function") {
      mq.addEventListener("change", mqHandler as any);
    } else {
      // older browsers
      // @ts-ignore
      mq.addListener?.(mqHandler);
    }
    const onResize = () => setIsPortrait(!!(window.matchMedia && window.matchMedia("(orientation: portrait)").matches));
    window.addEventListener("resize", onResize, { passive: true });
    setIsPortrait(!!(window.matchMedia && window.matchMedia("(orientation: portrait)").matches));
    return () => {
      if (typeof mq.removeEventListener === "function") {
        mq.removeEventListener("change", mqHandler as any);
      } else {
        // @ts-ignore
        mq.removeListener?.(mqHandler);
      }
      window.removeEventListener("resize", onResize);
    };
  }, []);

  // try to request fullscreen + lock orientation (best-effort)
  const tryLockLandscape = async () => {
    try {
      const docEl: any = document.documentElement;
      if (docEl.requestFullscreen) await docEl.requestFullscreen();
      // @ts-ignore
      const orientation = screen.orientation as any;
      if (orientation && typeof orientation.lock === "function") {
        try {
          await orientation.lock("landscape-primary");
        } catch {
          try {
            await orientation.lock("landscape");
          } catch {}
        }
      }
    } catch {
      // ignore failures
    }
  };

  // compute bounds for projection (from embedding coords)
  const bounds = useMemo(() => {
    if (!embeddings || embeddings.length === 0) return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const e of embeddings) {
      const x = Number(e.x), y = Number(e.y);
      if (Number.isFinite(x) && Number.isFinite(y)) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
    }
    if (minX === maxX) { minX -= 1; maxX += 1; }
    if (minY === maxY) { minY -= 1; maxY += 1; }
    return { minX, maxX, minY, maxY };
  }, [embeddings]);

  const project = (x: number, y: number) => {
    const { minX, maxX, minY, maxY } = bounds;
    const w = containerSize.w - 2 * padding;
    const h = containerSize.h - 2 * padding;
    const px = padding + ((x - minX) / (maxX - minX)) * w;
    const py = padding + (1 - (y - minY) / (maxY - minY)) * h;
    return { px, py };
  };

  function getCategoryIcon(cat: string) {
    switch (cat) {
      case "siren": return IconSiren;
      case "chirping_birds": return IconBird;
      case "breathing": return IconBreathing;
      case "washing_machine": return IconWashingMachine;
      case "crying_baby": return IconCryingBaby;
      default: return IconSiren;
    }
  }

  // tiny deterministic jitter
  const jitter = useMemo(() => {
    const xs = embeddings.map(e => Number(e.x));
    return 1e-3 * Math.max(1.0, (Math.max(...xs) - Math.min(...xs)));
  }, [embeddings]);

  // audio play / pause toggle
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

  // keep playing state synced on end / pause
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

  // pointer-based drag fallback (touch + mouse unified via Pointer Events)
  const draggingCatRef = useRef<string | null>(null);
  const pointerIdRef = useRef<number | null>(null);
  const ghostRef = useRef<HTMLElement | null>(null);

  const makeGhost = (cat: string, x: number, y: number) => {
    // remove old
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
            // perform synthetic drop
            setGuesses(prev => ({ ...prev, [file]: draggingCatRef.current as string }));
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
    try { ev.preventDefault(); ev.stopPropagation(); } catch {}
    try { (ev.currentTarget as Element).setPointerCapture?.(ev.pointerId); } catch {}
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

  // desktop drag handlers
  const onPaletteDragStart = (ev: React.DragEvent, category: string) => {
    try { ev.dataTransfer.setData("text/plain", category); } catch {}
  };

  const onPointDragOver = (ev: React.DragEvent) => ev.preventDefault();
  const onPointDrop = (ev: React.DragEvent, file: string) => {
    ev.preventDefault();
    const cat = ev.dataTransfer.getData("text/plain");
    if (cat && typeof cat === "string") setGuesses(prev => ({ ...prev, [file]: cat }));
  };

  const onPointDoubleClick = (file: string) => {
    setGuesses(prev => {
      const c = { ...prev };
      delete c[file];
      return c;
    });
  };

  const onCloseModal = () => setShowModal(false);
  const onResetGuesses = () => { setGuesses({}); setShowModal(false); };

  // rendering
  return (
    <div className="embed-viewer-root-fullscreen" style={{ position: "relative" }}>
      {/* orientation overlay (blocks interaction on portrait phones) */}
      {isPortrait && (
        <div className="landscape-overlay" role="dialog" aria-modal="true" aria-label="Rotate device to landscape">
          <div className="landscape-card" role="document">
            <h3 style={{ marginTop: 0 }}>Rotate your device</h3>
            <p style={{ margin: "8px 0 14px", color: "rgba(255,255,255,0.9)" }}>
              This game is best played in <strong>landscape</strong> orientation. Please rotate your phone.
            </p>
            <div style={{ display: "flex", gap: 10, justifyContent: "center", marginTop: 6 }}>
              <button onClick={() => tryLockLandscape()} className="landscape-btn" aria-label="Try fullscreen and rotate">
                Try fullscreen & rotate
              </button>
              <button onClick={() => setIsPortrait(false)} className="landscape-btn secondary" aria-label="Continue in portrait">
                Continue anyway
              </button>
            </div>
            <small style={{ display: "block", marginTop: 12, color: "rgba(255,255,255,0.6)" }}>
              Tip: If the button does nothing, rotate your device manually.
            </small>
          </div>
        </div>
      )}

      {/* lightweight top-left instructions */}
      <div
        style={{
          position: "absolute",
          top: 12,
          left: 12,
          zIndex: 1500,
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
        <div><strong>Note:</strong> Headphones recommended.</div>
      </div>

      {/* SVG canvas */}
      <svg width={containerSize.w} height={containerSize.h} className="embed-svg-full" style={{ display: "block" }}>
        <rect x={0} y={0} width={containerSize.w} height={containerSize.h} fill="#071030" />
        {embeddings.map((e, idx) => {
          const x = Number(e.x) + ((idx % 2 === 0 ? 1 : -1) * jitter * 0.5);
          const y = Number(e.y) + ((idx % 3 === 0 ? 1 : -1) * jitter * 0.5);
          const { px, py } = project(x, y);
          const isPlaying = playingFile === e.file;
          const guessed = guesses[e.file];
          const fillColor = guessed ? (CATEGORY_COLOR[guessed] ?? "#e0e0e0") : (isPlaying ? "#0b5c99" : "#ffffff");
          const strokeColor = guessed ? darken(CATEGORY_COLOR[guessed] ?? "#888") : (isPlaying ? "#0e4a7c" : "#6b7280");
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
              {/* only show native tooltip at the end to avoid revealing labels during play */}
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

      {/* palette (bottom-right) */}
      <div className="palette-root" aria-hidden style={{ touchAction: "none" }}>
        {PALETTE_CATEGORIES.map(cat => (
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

      {/* modal when done */}
      {showModal && (
        <div style={{
          position: "fixed", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
          background: "rgba(0,0,0,0.55)", zIndex: 2000
        }} role="dialog" aria-modal="true">
          <div style={{
            background: "#071030", color: "#fff", padding: 24, borderRadius: 10, minWidth: 320, maxWidth: "90%", textAlign: "center",
            boxShadow: "0 10px 40px rgba(0,0,0,0.6)"
          }}>
            <h2 style={{ margin: 0, marginBottom: 12 }}>Congratulations!</h2>
            <p style={{ margin: 0, marginBottom: 18, fontSize: 16 }}>
              Your final score: <strong style={{ fontSize: 18 }}>{correctCount} / {embeddings.length}</strong>
            </p>
            <div style={{ display: "flex", gap: 12, justifyContent: "center", marginTop: 8 }}>
              <button onClick={onCloseModal} style={{
                padding: "8px 14px", borderRadius: 6, border: "1px solid rgba(255,255,255,0.12)",
                background: "transparent", color: "#fff", cursor: "pointer"
              }}>Close</button>
              <button onClick={onResetGuesses} style={{
                padding: "8px 14px", borderRadius: 6, border: "none", background: "#ff6b6b", color: "#fff", cursor: "pointer"
              }}>Reset guesses</button>
            </div>
          </div>
        </div>
      )}

      <audio ref={audioRef} style={{ display: "none" }} preload="none" />
    </div>
  );
};

export default EmbeddingViewer;

// small helper
function darken(hex: string, amt = -24) {
  try {
    const parsed = hex.replace("#", "");
    const num = parseInt(parsed, 16);
    let r = (num >> 16) + amt;
    let g = ((num >> 8) & 0x00FF) + amt;
    let b = (num & 0x0000FF) + amt;
    r = Math.max(0, Math.min(255, r));
    g = Math.max(0, Math.min(255, g));
    b = Math.max(0, Math.min(255, b));
    return `rgb(${r},${g},${b})`;
  } catch {
    return hex;
  }
}
