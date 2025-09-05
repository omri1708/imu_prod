// ui_dsl/components/timeline.js
import { connectTimeline } from '../runtime/client.js';

export function mountTimeline(el, userId="anon") {
  const ul = document.createElement('ul');
  el.appendChild(ul);
  const stop = connectTimeline((ev)=>{
    const li = document.createElement('li');
    li.textContent = `[${new Date(ev.ts*1000).toISOString()}] ${ev.t} :: ${JSON.stringify(ev)}`;
    ul.appendChild(li);
  }, userId);
  return ()=>{ stop(); el.innerHTML=''; };
}