const canvas = document.querySelector('#orchard');
const ctx = canvas.getContext('2d');
let latest = null;
let lastEvent = -1;
const colors = {banana:'#f3c84b', flower:'#e796c5', rest:'#70c1a2'};
const labels = {banana:'香蕉', flower:'花朵', rest:'休息区'};

async function control(action) {
  const response = await fetch('/api/control', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({action})});
  latest = await response.json();
  render(latest);
}
for (const action of ['start','pause','step','reset']) document.querySelector(`#${action}`).addEventListener('click', () => control(action));

function scaleX(x, state) { return x / state.world.width * canvas.width; }
function scaleY(y, state) { return y / state.world.height * canvas.height; }

function draw(state) {
  const grad = ctx.createLinearGradient(0,0,canvas.width,canvas.height);
  grad.addColorStop(0,'#22382c'); grad.addColorStop(1,'#101d1a');
  ctx.fillStyle=grad; ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle='rgba(230,244,215,.035)'; ctx.lineWidth=1;
  for(let x=0;x<canvas.width;x+=48){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke();}
  for(let y=0;y<canvas.height;y+=48){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(canvas.width,y);ctx.stroke();}

  for(const patch of state.world.patches){
    const x=scaleX(patch.x,state), y=scaleY(patch.y,state), r=patch.radius/state.world.width*canvas.width;
    const halo=ctx.createRadialGradient(x,y,r*.2,x,y,155/state.world.width*canvas.width);
    halo.addColorStop(0,colors[patch.kind]+'55'); halo.addColorStop(1,colors[patch.kind]+'00');
    ctx.fillStyle=halo; ctx.beginPath(); ctx.arc(x,y,170,0,Math.PI*2); ctx.fill();
    ctx.fillStyle=colors[patch.kind]; ctx.globalAlpha=.88; ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fill(); ctx.globalAlpha=1;
    ctx.fillStyle='#0b1511'; ctx.font='700 12px ui-monospace'; ctx.textAlign='center'; ctx.fillText(labels[patch.kind],x,y+4);
  }
  const trail=state.fly.trail;
  if(trail.length>1){ctx.strokeStyle='rgba(226,244,184,.32)';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(scaleX(trail[0][0],state),scaleY(trail[0][1],state));for(const p of trail.slice(1))ctx.lineTo(scaleX(p[0],state),scaleY(p[1],state));ctx.stroke();}
  const x=scaleX(state.fly.x,state), y=scaleY(state.fly.y,state), a=state.fly.heading;
  ctx.save();ctx.translate(x,y);ctx.rotate(a);ctx.fillStyle='#f7f4df';ctx.beginPath();ctx.moveTo(13,0);ctx.lineTo(-9,-7);ctx.lineTo(-5,0);ctx.lineTo(-9,7);ctx.closePath();ctx.fill();ctx.fillStyle='rgba(200,230,107,.65)';ctx.beginPath();ctx.ellipse(-2,-8,8,4,-.35,0,Math.PI*2);ctx.ellipse(-2,8,8,4,.35,0,Math.PI*2);ctx.fill();ctx.restore();
}

function fmt(v, n=4){ return Number(v).toFixed(n); }
function bar(id,value,max){ document.querySelector(id).style.width=`${Math.min(100,Math.abs(value)/max*100)}%`; }
function render(state){
  if(!state || state.error) return;
  draw(state);
  document.querySelector('#statusText').textContent=state.running?'正在漫游':'已暂停';
  document.querySelector('#statusDot').classList.toggle('live',state.running);
  document.querySelector('#clock').textContent=`${state.sim_time.toFixed(1)} s · step ${state.step}`;
  document.querySelector('#seed').textContent=state.seed;
  document.querySelector('#model').textContent=state.neural.model;
  document.querySelector('#kc').textContent=fmt(state.neural.kc_activity,5); bar('#kcBar',state.neural.kc_activity,.08);
  document.querySelector('#apl').textContent=fmt(state.neural.apl_activity,5); bar('#aplBar',state.neural.apl_activity,.04);
  document.querySelector('#memory').textContent=fmt(state.neural.memory_norm,6); bar('#memoryBar',state.neural.memory_norm,.08);
  document.querySelector('#preferences').innerHTML=Object.entries(state.neural.preferences).map(([odor,value])=>`<div class="pref"><small>${labels[odor]}</small><strong style="color:${colors[odor]}">${value>=0?'+':''}${fmt(value,3)}</strong><small>innate ${fmt(state.neural.innate_preferences[odor],3)}<br>memory ${fmt(state.neural.learned_preferences[odor],4)}</small></div>`).join('');
  document.querySelector('#odors').innerHTML=Object.entries(state.odor).map(([odor,value])=>`<div class="odor"><span>${labels[odor]}</span><i><b style="width:${value*100}%;background:${colors[odor]}"></b></i><output>${fmt(value,3)}</output></div>`).join('');
  document.querySelector('#neuralTurn').textContent=fmt(state.control.neural_turn,3)+' rad/s';
  document.querySelector('#randomTurn').textContent=fmt(state.control.exploration_turn,3)+' rad/s';
  document.querySelector('#totalTurn').textContent=fmt(state.control.total_turn,3)+' rad/s';
  document.querySelector('#energy').textContent=fmt(state.fly.energy*100,1)+'%';
}

async function poll(){
  try { latest=await (await fetch('/api/state')).json(); render(latest); const data=await (await fetch(`/api/events?after=${lastEvent}`)).json(); if(data.events?.length){lastEvent=data.events.at(-1).sequence;const list=document.querySelector('#log');for(const event of data.events){const li=document.createElement('li');li.textContent=`${event.sim_time.toFixed(1)}s · ${event.kind}${event.payload.reward_dan?' · DAN reward':''}`;list.prepend(li);}while(list.children.length>12)list.lastChild.remove();}}
  catch(error){document.querySelector('#statusText').textContent='连接中断';}
  setTimeout(poll,250);
}
poll();
