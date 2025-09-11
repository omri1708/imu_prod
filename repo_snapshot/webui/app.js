(function(){
  const status = document.getElementById('status');
  const bar = document.getElementById('bar');
  const timeline = document.getElementById('timeline');
  const topic = 'run/default';
  const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/topic/' + encodeURIComponent(topic));

  function addEvent(ev){
    const div = document.createElement('div');
    div.className = 'event ' + (ev.ok ? 'ok' : 'fail');
    div.innerHTML = `<b>${ev.type || 'event'}</b> — ${ev.adapter || ''}<br/><code>${(ev.cmd||[]).join(' ')}</code>`;
    timeline.prepend(div);
    const cur = parseInt(bar.style.width || '0', 10);
    const next = Math.min(100, cur + 10);
    bar.style.width = next + '%';
    status.textContent = 'Live';
  }

  ws.onopen = () => { status.textContent = 'Connected'; };
  ws.onmessage = (m) => {
    try{
      const ev = JSON.parse(m.data);
      addEvent(ev);
    }catch(e){}
  };
  ws.onclose = () => { status.textContent = 'Disconnected'; };
})();