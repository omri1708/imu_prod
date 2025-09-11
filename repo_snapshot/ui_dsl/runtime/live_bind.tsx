// ui_dsl/runtime/live_bind.tsx
import React, { useEffect, useState } from "react";
import { StreamClient, StreamEvent } from "../components/streams";
import { ProgressBar, EventTimeline } from "../components/widgets";

export const LiveJobPane: React.FC<{wsUrl: string, topic: string}> = ({wsUrl, topic}) => {
  const [progress, setProgress] = useState(0);
  const [events, setEvents] = useState<StreamEvent[]>([]);
  useEffect(()=>{
    const sc = new StreamClient({url: `${wsUrl}?topic=${encodeURIComponent(topic)}`, burstLimit: 8, globalRatePerSec: 64});
    sc.connect((ev)=>{
      setEvents(prev => [ev, ...prev].slice(0, 200));
      if (ev.type === "progress" && typeof ev.payload?.pct === "number") {
        setProgress(ev.payload.pct);
      }
    });
  }, [wsUrl, topic]);
  return (
    <div style={{display:"flex", gap: 16}}>
      <ProgressBar progress={progress} label="Build/Deploy" />
      <EventTimeline events={events} />
    </div>
  );
};