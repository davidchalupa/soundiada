// src/components/EmbeddingViewer.tsx
import React, { useRef, useState, useMemo, useEffect } from "react";
import { esc50Embeddings } from "../resources/embeddings";
import seedrandom from "seedrandom";
import "./EmbeddingViewer.css";

type Esc50Embedding = {
  file: string;
  category: string;
  x: number;
  y: number;
};

// five-category palette (from your sample)
const PALETTE_CATEGORIES = [
  "siren",
  "chirping_birds",
  "breathing",
  "washing_machine",
  "crying_baby",
] as const;
type PaletteCat = typeof PALETTE_CATEGORIES[number];

// color map for categories (tweak as you like)
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
  const containerRef = useRef<HTMLDivElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playingFile, setPlayingFile] = useState<string | null>(null);
  const [containerSize, setContainerSize] = useState({ w: window.innerWidth, h: window.innerHeight });

  // guesses: mapping file -> guessed category
  const [guesses, setGuesses] = useState<Record<string, string>>({});

  // frozen copy of embeddings
  const embeddings = useMemo(() => esc50Embeddings.slice(), []);

  // compute bounds
  const bounds = useMemo(() => {
    if (!embeddings || embeddings.length === 0) {
      return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
    }
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

  // project to pixels inside box
  const project = (x: number, y: number) => {
    const { minX, maxX, minY, maxY } = bounds;
    const w = containerSize.w - 2 * padding;
    const h = containerSize.h - 2 * padding;
    const px = padding + ((x - minX) / (maxX - minX)) * w;
    const py = padding + (1 - (y - minY) / (maxY - minY)) * h;
    return { px, py };
  };

  // responsive size
  // responsive size (stable: use window resize instead of ResizeObserver to avoid layout loops)
  useEffect(() => {
    const update = () => {
      setContainerSize({
        w: Math.max(200, window.innerWidth),
        h: Math.max(160, window.innerHeight),
      });
    };
    // initialize once
    update();
    // update on window resize
    window.addEventListener("resize", update, { passive: true });
    return () => {
      window.removeEventListener("resize", update);
    };
  }, []);


  // deterministic tiny jitter (keeps overlap visible)
  const jitter = 1e-3 * Math.max(1.0, (Math.max(...embeddings.map(e => Number(e.x))) - Math.min(...embeddings.map(e => Number(e.x)))));
  const rng = useMemo(() => seedrandom("42"), []);

  // Play / Pause toggle
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
      if (a.currentTime > 0 && !a.ended) {
        setPlayingFile(null);
      }
    };
    a.addEventListener("ended", onEnded);
    a.addEventListener("pause", onPause);
    return () => {
      a.removeEventListener("ended", onEnded);
      a.removeEventListener("pause", onPause);
    };
  }, []);

  // Drag handlers for palette
  const onPaletteDragStart = (ev: React.DragEvent, category: string) => {
    try {
      ev.dataTransfer.setData("text/plain", category);
      // optionally set drag image - omitted for simplicity
    } catch (e) {
      // ignore
    }
  };

  // Drop handlers for points
  const onPointDragOver = (ev: React.DragEvent) => {
    ev.preventDefault(); // allow drop
  };

  const onPointDrop = (ev: React.DragEvent, file: string) => {
    ev.preventDefault();
    const cat = ev.dataTransfer.getData("text/plain");
    if (cat && typeof cat === "string") {
      setGuesses(prev => ({ ...prev, [file]: cat }));
    }
  };

  // Optional: ability to clear guess by right-click (contextmenu) or double-click
  const onPointDoubleClick = (file: string) => {
    // remove guess
    setGuesses(prev => {
      const copy = { ...prev };
      delete copy[file];
      return copy;
    });
  };

  // render
  return (
    <div ref={containerRef} className="embed-viewer-root-fullscreen">
      <svg width={containerSize.w} height={containerSize.h} className="embed-svg-full">
        <rect x="0" y="0" width={containerSize.w} height={containerSize.h} fill="#071030" />
        {embeddings.map((e, idx) => {
          const x = Number(e.x) + ((idx % 2 === 0 ? 1 : -1) * jitter * 0.5);
          const y = Number(e.y) + ((idx % 3 === 0 ? 1 : -1) * jitter * 0.5);
          const { px, py } = project(x, y);
          const isPlaying = playingFile === e.file;
          const guessed = guesses[e.file]; // may be undefined
          const fillColor = guessed ? (CATEGORY_COLOR[guessed] ?? "#e0e0e0") : (isPlaying ? "#0b5c99" : "#ffffff");
          const strokeColor = guessed ? darken(CATEGORY_COLOR[guessed] ?? "#888") : (isPlaying ? "#0e4a7c" : "#6b7280");
          const key = `${e.file}-${idx}`;
          return (
            <g
              key={key}
              transform={`translate(${px}, ${py})`}
              className="embed-point"
              style={{ cursor: "pointer" }}
              onClick={() => onPointClick(e.file)}
              onDoubleClick={() => onPointDoubleClick(e.file)}
              onDragOver={onPointDragOver}
              onDrop={(ev) => onPointDrop(ev, e.file)}
              title={`${e.file} — actual: ${e.category} — guessed: ${guessed ?? "none"}`}
            >
              <circle r={16} fill="rgba(255,255,255,0.02)" stroke="none" />
              <circle r={12} fill={fillColor} stroke={strokeColor} strokeWidth={isPlaying ? 2.2 : 1.0} />
              {guessed ? (
                // small marker showing guessed category initial
                <text x={-3} y={4} fontSize={7} fill="#fff" style={{ pointerEvents: "none" }}>{shortLabel(guessed)}</text>
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

      {/* palette: draggable category icons (lower-right) */}
      <div className="palette-root" aria-hidden>
        {PALETTE_CATEGORIES.map((cat) => (
          <div
            key={cat}
            className="palette-item"
            draggable
            onDragStart={(ev) => onPaletteDragStart(ev, cat)}
            title={`Drag to assign category: ${cat}`}
          >
            <div className="palette-dot" style={{ background: CATEGORY_COLOR[cat] }} />
            <div className="palette-label">{shortLabel(cat)}</div>
          </div>
        ))}
      </div>

      <audio ref={audioRef} style={{ display: "none" }} preload="none" />
    </div>
  );
};

export default EmbeddingViewer;

/* small helper functions */
function shortLabel(cat: string) {
  // returns a short label to render inside the node / palette (2-3 chars)
  if (!cat) return "";
  const parts = cat.split(/[_-]/);
  if (parts.length === 1) return cat.slice(0, 3).toUpperCase();
  return parts.map(p => p[0].toUpperCase()).join("").slice(0,3);
}
function darken(hex: string, amt = -24) {
  // tiny hex darken helper, amt negative to darken
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
