// ui_dsl/components/widgets.tsx
import React from "react";
import { StreamEvent } from "./streams";

export const ProgressBar: React.FC<{progress: number, label?: string}> = ({progress, label}) => {
  const pct = Math.max(0, Math.min(100, progress));
  return (
    <div style={{border: "1px solid #aaa", borderRadius: 6, padding: 4, width: 320}}>
      <div style={{fontSize: 12, marginBottom: 4}}>{label ?? "Progress"}</div>
      <div style={{background: "#eee", height: 12, borderRadius: 6, overflow: "hidden"}}>
        <div style={{width: `${pct}%`, height: "100%"}} />
      </div>
      <div style={{fontSize: 12, marginTop: 4}}>{pct.toFixed(1)}%</div>
    </div>
  );
};

export const EventTimeline: React.FC<{events: StreamEvent[]}> = ({events}) => {
  return (
    <div style={{border: "1px solid #ddd", borderRadius: 6, padding: 8, maxHeight: 260, overflowY: "auto", width: 480}}>
      {events.map((e, i) => (
        <div key={i} style={{display: "flex", gap: 8, marginBottom: 6}}>
          <div style={{fontFamily: "monospace", fontSize: 12, color: "#666"}}>{new Date(e.ts).toLocaleTimeString()}</div>
          <div style={{fontWeight: 600}}>{e.type}</div>
          <div style={{whiteSpace: "pre-wrap"}}>{JSON.stringify(e.payload)}</div>
        </div>
      ))}
    </div>
  );
};