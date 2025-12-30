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

/* build audio URL (raw GitHub) */
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

  // resize observer to keep full-screen responsive
  useEffect(() => {
    const el = containerRef.current ?? document.documentElement;
    const ro = new ResizeObserver(() => {
      const rect = (containerRef.current ?? document.documentElement).getBoundingClientRect();
      setContainerSize({ w: Math.max(200, rect.width), h: Math.max(160, rect.height) });
    });
    ro.observe(el);
    // init
    const rect = (containerRef.current ?? document.documentElement).getBoundingClientRect();
    setContainerSize({ w: Math.max(200, rect.width || window.innerWidth), h: Math.max(160, rect.height || window.innerHeight) });
    return () => ro.disconnect();
  }, []);

  // deterministic tiny jitter (not required, but keeps overlap visible)
  const jitter = 1e-3 * Math.max(1.0, (Math.max(...embeddings.map(e => Number(e.x))) - Math.min(...embeddings.map(e => Number(e.x)))));
  const rng = useMemo(() => seedrandom("42"), []);

  // Play / Pause toggle behavior:
  // - if clicked file is currently playing -> pause it
  // - if clicked file is same but paused -> resume
  // - otherwise stop current and play clicked
  const onPointClick = (file: string) => {
    const url = buildAudioUrl(file);
    const audio = audioRef.current;
    if (!audio) return;

    const currentSrcHasFile = audio.src && audio.src.includes(file);

    // If same file is playing -> pause
    if (currentSrcHasFile && !audio.paused) {
      audio.pause();
      setPlayingFile(null);
      return;
    }

    // If same file is loaded but paused -> resume
    if (currentSrcHasFile && audio.paused) {
      audio.play().then(() => setPlayingFile(file)).catch(() => setPlayingFile(null));
      return;
    }

    // Otherwise load new file and play
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

  // when audio ends, clear playing state
  useEffect(() => {
    const a = audioRef.current;
    if (!a) return;
    const onEnded = () => setPlayingFile(null);
    a.addEventListener("ended", onEnded);
    // when paused via external controls, we also clear playing state
    const onPause = () => {
      if (a.currentTime > 0 && !a.ended) {
        // treat user pause as stop highlight
        setPlayingFile(null);
      }
    };
    a.addEventListener("pause", onPause);
    return () => {
      a.removeEventListener("ended", onEnded);
      a.removeEventListener("pause", onPause);
    };
  }, []);

  return (
    <div ref={containerRef} className="embed-viewer-root-fullscreen">
      <svg width={containerSize.w} height={containerSize.h} className="embed-svg-full">
        {/* background */}
        <rect x="0" y="0" width={containerSize.w} height={containerSize.h} fill="#071030" />

        {embeddings.map((e, idx) => {
          const x = Number(e.x) + ((idx % 2 === 0 ? 1 : -1) * jitter * 0.5);
          const y = Number(e.y) + ((idx % 3 === 0 ? 1 : -1) * jitter * 0.5);
          const { px, py } = project(x, y);
          const isPlaying = playingFile === e.file;
          const key = `${e.file}-${idx}`;

          return (
            <g
              key={key}
              transform={`translate(${px}, ${py})`}
              className="embed-point"
              style={{ cursor: "pointer" }}
              onClick={() => onPointClick(e.file)}
            >
              {/* subtle halo */}
              <circle r={16} fill="rgba(255,255,255,0.02)" stroke="none" />
              {/* main button */}
              <circle r={12} fill={isPlaying ? "#0b5c99" : "#ffffff"} stroke={isPlaying ? "#0e4a7c" : "#6b7280"} strokeWidth={isPlaying ? 2.2 : 1.0} />
              {/* icon: pause (two rects) when playing, triangle otherwise */}
              {isPlaying ? (
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

      {/* hidden audio element used for playback */}
      <audio ref={audioRef} style={{ display: "none" }} preload="none" />

    </div>
  );
};

export default EmbeddingViewer;
