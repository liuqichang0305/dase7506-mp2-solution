/** Run the unchanged published peer-review handler offline. Never publishes. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL, fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {setTimeout as delay} from 'node:timers/promises';

const hash = x => createHash('sha256').update(x).digest('hex');
const requireThat = (condition, message) => { if (!condition) throw new Error(message); };
const descendants = node => [node, ...node.children.flatMap(x => typeof x === 'object' ? descendants(x) : [])];
class Element {
  constructor(tag='div', id='') {
    this.tagName=tag.toUpperCase(); this.id=id; this.children=[]; this.listeners=new Map();
    this.value=''; this.checked=false; this.disabled=false; this.required=false; this.hidden=false;
    this.textContent=''; this.dataset={}; this.classList={toggle(){}};
  }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children=items; this.textContent=''; }
  addEventListener(type, callback) { this.listeners.set(type,[...(this.listeners.get(type)||[]),callback]); }
  setAttribute(name,value) { this[name]=String(value); }
  removeAttribute(name) { delete this[name]; }
  reset() { this.wasReset=true; }
  showModal() { this.shown=true; }
  close() { this.shown=false; }
  reportValidity() { return true; }
}
async function inside(root, relative) {
  const resolved=path.resolve(root,relative), rel=path.relative(root,resolved);
  requireThat(rel && !rel.startsWith('..') && !path.isAbsolute(rel), 'Path leaves repository scope');
  const actual=await fs.realpath(resolved);
  requireThat(actual===resolved, 'Symlinked input is not accepted');
  return resolved;
}
async function checked(root, item) {
  const full=await inside(root,item.path), raw=await fs.readFile(full);
  requireThat(hash(raw)===item.sha256, 'Input/source hash changed: '+item.path);
  return {full,raw};
}

export async function prepare({root,inputPath,outputDir}) {
  root=await fs.realpath(root);
  const inputBytes=await fs.readFile(await inside(root,inputPath));
  const input=JSON.parse(inputBytes);
  requireThat(input.schema==='mp2-peer-report-input-v1' && input.intended_account==='liuqichang0305', 'Wrong preparation scope');
  for(const [name,digest] of Object.entries(input.input_sha256)) await checked(root,{path:name,sha256:digest});
  for(const [name,digest] of Object.entries(input.tool_sources)) await checked(root,{path:name,sha256:digest});
  const snapshotFile=await checked(root,input.official_snapshot);
  const snapshot=JSON.parse(snapshotFile.raw);
  requireThat(snapshot.schema==='mp2-peer-official-form-source-v1', 'Wrong source manifest');
  const official=new Map();
  for(const item of snapshot.files) {
    const value=await checked(root,{path:snapshot.source_root+'/'+item.path,sha256:item.sha256});
    requireThat(value.raw.length===item.bytes, 'Official source length changed');
    official.set(item.path,value);
  }
  const html=official.get(snapshot.published_html)?.raw.toString('utf8');
  requireThat(html && html.includes('src="../assets/'+path.posix.basename(snapshot.published_entry)+'"'), 'HTML entry graph differs');
  requireThat(/id="reproduced-score"[^>]*required/.test(html), 'Official required reproduced-score field missing');
  requireThat(/id="review-reason"[^>]*maxlength="2000"/.test(html), 'Official reason limit differs');
  const entry=official.get(snapshot.published_entry), arenaFile=official.get(snapshot.published_arena);
  requireThat(entry && arenaFile && entry.raw.toString().includes("'./"+path.posix.basename(snapshot.published_arena)+"'"), 'Entry arena import differs');
  requireThat(arenaFile.raw.toString().includes("'./"+path.posix.basename(snapshot.published_projects)+"'"), 'Arena projects import differs');
  const arena=await import(pathToFileURL(arenaFile.full).href);
  const publication={projects:{mp2:{published_at:input.window.published_at,review_ends_at:input.window.review_ends_at}}};
  requireThat(arena.submissionPhase('mp2',publication)==='review', 'Official peer-review window is not open');
  requireThat(arena.REPO==='xudongwu-0/xudongwu-0.github.io', 'Wrong report destination');
  const target=input.target, fields=input.fields;
  requireThat(Number.isSafeInteger(target.number) && target.number>0 && target.project==='mp2' && target.author!=='liuqichang0305', 'Invalid/self target');
  requireThat(arena.validScore('mp2',fields.reproduced_score), 'Invalid reproduced score');
  requireThat(typeof fields.reason==='string' && fields.reason.trim()===fields.reason && fields.reason.length<=2000, 'Invalid optional Markdown');
  requireThat(fields.evidence_url===null || (!!arena.safeURL(fields.evidence_url) && fields.evidence_url.length<=500), 'Invalid optional evidence URL');
  // Only the independently selected target is supplied. No board is downloaded.
  const board={schema:'dase7506/board-v1',snapshot_version:2,protocol:arena.PROTOCOL,
    generated_at:new Date().toISOString(),submissions:[target],challenges:[],adjustments:[],publication};
  const nodes=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(m=>[m[1],new Element('div',m[1])]));
  const saved={}; for(const name of ['document','window','fetch','setInterval']) saved[name]=Object.getOwnPropertyDescriptor(globalThis,name);
  const fetchLog=[], intervals=[];
  const set=(name,value)=>Object.defineProperty(globalThis,name,{value,writable:true,configurable:true});
  set('document',{hidden:false,getElementById(id){requireThat(nodes.has(id),'Unknown official DOM field: '+id);return nodes.get(id);},
    createElement(tag){return new Element(tag);},querySelectorAll(selector){
      if(selector==='[data-project]') return [];
      if(selector==='.link-requirement') return [new Element('span'),new Element('span')];
      throw new Error('Unexpected selector: '+selector);
    },addEventListener(){}});
  set('window',{isSecureContext:true,addEventListener(){}});
  set('setInterval',(callback,ms)=>{intervals.push(ms);return intervals.length;});
  set('fetch',async (url,options={})=>{
    const value=String(url);
    requireThat((!options.method||options.method==='GET') &&
      (value==='../data/leaderboard.json' || value.startsWith('https://raw.githubusercontent.com/'+arena.REPO+'/main/courses/dase7506/data/leaderboard.json?v=') ||
       value==='https://api.github.com/repos/'+arena.REPO+'/contents/courses/dase7506/data/leaderboard.json?ref=main'), 'Unexpected fetch blocked');
    fetchLog.push({requested_url:value,served:'one explicitly selected target plus publication dates',network_used:false});
    return new Response(JSON.stringify(board),{status:200,headers:{'Content-Type':'application/json'}});
  });
  let generated;
  try {
    await import(pathToFileURL(entry.full).href+'?peer_adapter='+hash(inputBytes));
    const stop=Date.now()+3000;
    while(!nodes.get('sync-status').textContent.startsWith('Updated:') && Date.now()<stop) await delay(10);
    requireThat(nodes.get('sync-status').textContent.startsWith('Updated:'),'Official refresh did not accept the single-target board');
    requireThat(nodes.get('phase-badge').textContent==='7-day public review' && nodes.get('submit').hidden, 'Official form is not in review phase');
    const buttons=descendants(nodes.get('board-body')).filter(n=>n.tagName==='BUTTON' && n.textContent==='Peer Review Report');
    requireThat(buttons.length===1 && !buttons[0].disabled, 'Target report button is missing or disabled');
    const click=buttons[0].listeners.get('click')||[]; requireThat(click.length===1,'Unexpected target handler count'); click[0]();
    requireThat(nodes.get('challenge-dialog').shown,'Official target dialog did not open');
    nodes.get('reproduced-score').value=String(fields.reproduced_score);
    nodes.get('review-reason').value=fields.reason;
    nodes.get('evidence-url').value=fields.evidence_url||'';
    const handlers=nodes.get('challenge-form').listeners.get('submit')||[];
    requireThat(handlers.length===1,'Exactly one official peer submit handler required');
    let prevented=false; handlers[0]({preventDefault(){prevented=true;}});
    requireThat(prevented,'Official handler did not prevent navigation');
    const status=descendants(nodes.get('challenge-status'));
    requireThat(!status.some(n=>n.className==='error'),'Official handler rejected fields: '+status.filter(n=>n.className==='error').map(n=>n.textContent).join('; '));
    const links=status.filter(n=>n.tagName==='A' && n.href);
    requireThat(links.length===1,'Official handler emitted no unique issue URL');
    const url=new URL(links[0].href), title=url.searchParams.get('title'), body=url.searchParams.get('body');
    requireThat(url.origin==='https://github.com' && url.pathname==='/'+arena.REPO+'/issues/new','Wrong issue destination');
    const blocks=[...body.matchAll(/```json\s*\n([\s\S]*?)\n```/g)]; requireThat(blocks.length===1,'Expected one official JSON block');
    const payload=JSON.parse(blocks[0][1]);
    const expected={schema:'dase7506/peer-review-v1',protocol:arena.PROTOCOL,project:'mp2',submission:target.number,
      reason:fields.reason,reproduced_score:fields.reproduced_score,evidence_url:fields.evidence_url};
    requireThat(JSON.stringify(payload)===JSON.stringify(expected) && arena.validatePeerReviewReport(payload),'Official payload differs from bound fields');
    requireThat(arena.makeIssueURL(payload)===url.href,'Official formatter round trip differs');
    requireThat(arena.submissionPhase('mp2',publication)==='review','Review window ended during preparation');
    generated={schema:'mp2-peer-generated-form-v1',status:'prepared_not_submitted',generated_at_utc:new Date().toISOString(),
      intended_account:input.intended_account,expected_author_type:'User',target_submission:target.number,
      official_source_commit:snapshot.official_site_commit,official_source_snapshot_sha256:hash(snapshotFile.raw),
      official_executed_entry:snapshot.published_entry,report_input_sha256:hash(inputBytes),
      official_payload:payload,issue_title:title,issue_body:body,issue_url:url.href,
      issue_body_sha256:hash(body),issue_title_sha256:hash(title),issue_url_sha256:hash(url.href),
      official_phase:'review',official_phase_badge:nodes.get('phase-badge').textContent,
      captured_publication:publication,visible_target:nodes.get('challenge-target').textContent,
      adapter_sha256:hash(await fs.readFile(fileURLToPath(import.meta.url))),fetch_log:fetchLog,intervals_suppressed:intervals,
      scope:'Unchanged published peer handler and exact dependencies. Single-target local DOM; no network, encrypted artifact operation, score submission, email or issue creation.'};
  } finally {
    for(const [name,descriptor] of Object.entries(saved)) if(descriptor) Object.defineProperty(globalThis,name,descriptor); else delete globalThis[name];
  }
  for(const [name,digest] of Object.entries(input.input_sha256)) await checked(root,{path:name,sha256:digest});
  const out=path.resolve(root,outputDir);
  requireThat(out.startsWith(root+path.sep), 'Output leaves root');
  await fs.mkdir(out,{recursive:true});
  for(const [name,content] of [['GENERATED_FORM.json',JSON.stringify(generated,null,2)+'\n'],['issue-title.txt',generated.issue_title],['issue-body.txt',generated.issue_body],['issue-url.txt',generated.issue_url]])
    await fs.writeFile(path.join(out,name),content,{encoding:'utf8',flag:'wx'});
  return generated;
}
if(process.argv[1] && path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  const args=process.argv.slice(2); const opts={};
  for(let i=0;i<args.length;i+=2) { requireThat(['--root','--input','--output'].includes(args[i]) && args[i+1] && !opts[args[i]],'Invalid arguments');opts[args[i]]=args[i+1]; }
  prepare({root:opts['--root'],inputPath:opts['--input'],outputDir:opts['--output']}).then(result=>console.log(JSON.stringify({status:result.status,target:result.target_submission})))
    .catch(error=>{console.error(error.message);process.exitCode=1;});
}
