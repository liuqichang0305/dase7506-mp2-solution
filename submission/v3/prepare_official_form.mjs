/**
 * Execute the unchanged, actually published official MP2 submit handler locally.
 * This prepares files only. It cannot publish an issue or make a network request.
 * Native browser form validation is explicitly mirrored for the supplied fields;
 * payload construction, encryption, validation and issue formatting are official code.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {createHash, webcrypto} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {setTimeout as delay} from 'node:timers/promises';
import {createArtifactObserver, installCryptoOverride} from './artifact_binding.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const hash = value => createHash('sha256').update(value).digest('hex');
const declaration = 'I understand and confirm that this is my result.';
const account = 'liuqichang0305';
const repository = `${account}/dase7506-mp2-solution`;
function assert(condition, message) { if (!condition) throw new Error(message); }
function optionArguments(args) {
  const parsed = {};
  for (let i=0; i<args.length; i++) {
    const name=args[i];
    assert(name.startsWith('--'), `Unexpected argument: ${name}`);
    assert(!Object.hasOwn(parsed,name), `Repeated argument: ${name}`);
    if (name==='--confirm-result' || name==='--help') parsed[name]=true;
    else {
      assert(i+1<args.length && !args[i+1].startsWith('--'), `Missing value for ${name}`);
      parsed[name]=args[++i];
    }
  }
  const allowed=new Set(['--score','--code-url','--checkpoint-url','--student-id','--confirm-result','--source-dir','--freeze','--output-dir','--help']);
  for(const name of Object.keys(parsed)) assert(allowed.has(name),`Unknown argument: ${name}`);
  return parsed;
}
function localPath(value) { return path.resolve(root,value); }
function inside(base, relative) {
  const full=path.resolve(base,relative), rel=path.relative(base,full);
  assert(rel && !rel.startsWith('..') && !path.isAbsolute(rel),`Unsafe relative path: ${relative}`);
  return full;
}
function immutableGithubLink(value, kind) {
  const url=new URL(value);
  assert(url.protocol==='https:' && !url.username && !url.password && !url.search && !url.hash,'Expected clean HTTPS GitHub artifact link');
  assert(url.hostname==='github.com', 'Use the verified public GitHub repository links');
  const prefix=`/${repository}/tree/`;
  assert(url.pathname.startsWith(prefix),'Artifact link uses a different repository or non-immutable ref');
  const rest=url.pathname.slice(prefix.length).replace(/\/$/,'');
  const [commit,...suffix]=rest.split('/');
  assert(/^[0-9a-f]{40}$/.test(commit),'Artifact links must identify a full immutable commit SHA');
  assert(suffix.join('/')===(kind==='code'?'':'artifacts/research_v3/final_selected'),`Unexpected ${kind} artifact path`);
  return commit;
}

class Element {
  constructor(tag='div',id='') {
    this.tagName=tag.toUpperCase(); this.id=id; this.children=[]; this.listeners=new Map();
    this.value=''; this.checked=false; this.disabled=false; this.required=false;
    this.hidden=false; this.dataset={}; this.textContent=''; this.attributes={};
  }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children=[...nodes]; this.textContent=''; }
  addEventListener(type,callback) {
    if(!this.listeners.has(type)) this.listeners.set(type,[]);
    this.listeners.get(type).push(callback);
  }
  setAttribute(key,value) { this.attributes[key]=String(value); }
  removeAttribute(key) { delete this.attributes[key]; }
  reset() { throw new Error('Peer-review form is out of scope'); }
  showModal() { throw new Error('Dialog interaction is out of scope'); }
  close() { throw new Error('Dialog interaction is out of scope'); }
  reportValidity() { throw new Error('Unexpected browser-only validation path'); }
}
function descendants(node) { return [node,...node.children.flatMap(child=>typeof child==='object'?descendants(child):[])]; }

async function main() {
  const args=optionArguments(process.argv.slice(2));
  if(args['--help']) {
    console.log('Prepare only; does not create an issue.\nnode submission/v3/prepare_official_form.mjs --score submission/v3/test-score.json --student-id 3036900957 --code-url https://github.com/liuqichang0305/dase7506-mp2-solution/tree/FULL_COMMIT --checkpoint-url https://github.com/liuqichang0305/dase7506-mp2-solution/tree/FULL_COMMIT/artifacts/research_v3/final_selected --confirm-result');
    return;
  }
  for(const name of ['--score','--code-url','--checkpoint-url','--student-id']) assert(args[name],`Required: ${name}`);
  assert(args['--student-id']==='3036900957','Student identity must match this explicitly authorized project');
  assert(args['--confirm-result']===true,'Explicit --confirm-result is required for the authorized result declaration');
  const sourceDir=localPath(args['--source-dir']||'tmp/form-source-v3');
  const outputDir=localPath(args['--output-dir']||'submission/v3');
  const freezePath=localPath(args['--freeze']||'submission/v3/FREEZE.json');
  const scorePath=localPath(args['--score']);
  const outputs=['ARTIFACT_BINDING.json','GENERATED_FORM.json','issue-title.txt','issue-body.txt','issue-url.txt'];
  await fs.mkdir(outputDir,{recursive:true});
  for(const name of outputs) {
    try { await fs.access(path.join(outputDir,name)); throw new Error(`Refusing to overwrite ${name}`); }
    catch(error) { if(error.code!=='ENOENT') throw error; }
  }
  const startedAt=new Date().toISOString();
  const snapshotBytes=await fs.readFile(root+'/submission/v3/FORM_SOURCE_SNAPSHOT.json');
  const snapshot=JSON.parse(snapshotBytes);
  assert(/^[0-9a-f]{40}$/.test(snapshot.official_site_commit),'Missing official form source commit');
  assert(snapshot.published_entry_path && snapshot.published_module_graph,'The actual page-referenced module graph is required');
  const verified=[];
  for(const item of snapshot.files) {
    const local=inside(sourceDir,item.local_path||path.posix.basename(item.path));
    const bytes=await fs.readFile(local);
    assert(hash(bytes)===item.sha256 && bytes.length===item.bytes,`Official source snapshot changed: ${item.path}`);
    verified.push({...item,absolute_local_path:local});
  }
  const findOfficial=sourcePath=>{
    const found=verified.find(item=>item.path===sourcePath);
    assert(found,`Missing verified official source ${sourcePath}`);
    return found.absolute_local_path;
  };
  const entry=findOfficial(snapshot.published_entry_path);
  const arenaPath=snapshot.published_module_graph[snapshot.published_entry_path].find(p=>/\/arena\.[0-9a-f]+\.mjs$/.test(p));
  assert(arenaPath,'Published form has no reviewed arena module import');
  const arena=await import(pathToFileURL(findOfficial(arenaPath)).href);
  const html=await fs.readFile(findOfficial('courses/dase7506/mp2/index.html'),'utf8');
  assert(html.includes(declaration),'Official declaration changed; review current HTML before preparing');
  assert(/id="consent"[^>]*type="checkbox"[^>]*required/.test(html),'Expected required result-declaration checkbox');
  const freezeBytes=await fs.readFile(freezePath), freeze=JSON.parse(freezeBytes);
  assert(freeze.frozen===true && freeze.test_evaluated_before_freeze===false,'Missing valid pre-test freeze');
  assert(/^[0-9a-f]{40}$/.test(freeze.source_commit),'Freeze source commit missing');
  const codeCommit=immutableGithubLink(args['--code-url'],'code');
  const checkpointCommit=immutableGithubLink(args['--checkpoint-url'],'checkpoint');
  assert(codeCommit===checkpointCommit,'Code and checkpoint links must use the same final immutable commit');
  execFileSync('git',['merge-base','--is-ancestor',freeze.source_commit,codeCommit],{cwd:root,stdio:['ignore','pipe','pipe']});
  const frozenFiles={...freeze.source_files,...freeze.assets};
  for(const [relative,expected] of Object.entries(frozenFiles)) {
    assert(hash(await fs.readFile(inside(root,relative)))===expected,`Local frozen file changed: ${relative}`);
    const committed=execFileSync('git',['show',`${codeCommit}:${relative}`],{cwd:root,maxBuffer:20*1024*1024,stdio:['ignore','pipe','pipe']});
    assert(hash(committed)===expected,`Published commit does not match frozen file: ${relative}`);
  }
  const scoreBytes=await fs.readFile(scorePath), score=JSON.parse(scoreBytes);
  const dataManifest=JSON.parse(await fs.readFile(root+'/code/data/manifest.json','utf8'));
  assert(score.protocol===dataManifest.protocol && score.split==='test','Use the official frozen full-test score JSON only');
  assert(Number.isSafeInteger(score.n) && score.n===dataManifest.splits.test,'Score must cover the complete official test bank');
  assert(Number.isSafeInteger(score.correct) && score.correct>=0 && score.correct<=score.n,'Invalid official correct count');
  assert(Number.isFinite(score.accuracy) && Math.abs(score.accuracy-100*score.correct/score.n)<1e-12,'Accuracy disagrees with the official correct/n count');
  assert((await fs.stat(scorePath)).mtimeMs>=Date.parse(freeze.created_at_utc),'Score file predates the pre-test freeze');
  const board=JSON.parse(await fs.readFile(findOfficial('courses/dase7506/data/leaderboard.json'),'utf8'));
  assert(board.schema==='dase7506/board-v1' && board.snapshot_version===2 && board.protocol===arena.PROTOCOL,'Unexpected official board schema');
  assert(arena.submissionPhase('mp2',board.publication)==='scores','Official current-time phase is not scores; do not generate a new-score submission');
  assert(arena.safeURL(args['--code-url']) && arena.safeURL(args['--checkpoint-url']),'Official URL validation failed');
  const observer=createArtifactObserver(globalThis.crypto?.subtle ? globalThis.crypto : webcrypto,
    {project:'mp2',code_url:args['--code-url'],checkpoint_url:args['--checkpoint-url']});
  const observerProvenance={
    adapter_sha256:hash(await fs.readFile(fileURLToPath(import.meta.url))),
    observer_sha256:hash(await fs.readFile(fileURLToPath(new URL('./artifact_binding.mjs',import.meta.url)))),
    official_entry_sha256:hash(await fs.readFile(entry)),
  };

  const nodes=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(match=>[match[1],new Element('div',match[1])]));
  const linkRequirements=[new Element('span'),new Element('span')];
  const documentListeners=[];
  const windowListeners=[];
  const timerRegistrations=[];
  const fetchLog=[];
  const preserved={document:globalThis.document,window:globalThis.window,fetch:globalThis.fetch,setInterval:globalThis.setInterval};
  globalThis.document={hidden:false,
    getElementById(id) { assert(nodes.has(id),`Official handler requested unknown HTML ID: ${id}`); return nodes.get(id); },
    createElement(tag) { return new Element(tag); },
    querySelectorAll(selector) {
      if(selector==='[data-project]') return [];
      if(selector==='.link-requirement') return linkRequirements;
      throw new Error(`Unreviewed DOM selector: ${selector}`);
    },
    addEventListener(type,callback) { documentListeners.push({type,callback}); },
  };
  globalThis.window={isSecureContext:true,addEventListener(type,callback) {windowListeners.push({type,callback});}};
  globalThis.setInterval=(callback,milliseconds)=>{timerRegistrations.push({callback,milliseconds});return timerRegistrations.length;};
  globalThis.fetch=async (url,options={})=>{
    assert(!options.method || options.method==='GET','Only source-snapshot GET reads are supported');
    const value=String(url);
    let source;
    if(value==='../data/student-id-public.pem') source='courses/dase7506/data/student-id-public.pem';
    else if(value==='../data/leaderboard.json' || value.startsWith(`https://raw.githubusercontent.com/${arena.REPO}/main/courses/dase7506/data/leaderboard.json?v=`) || value===`https://api.github.com/repos/${arena.REPO}/contents/courses/dase7506/data/leaderboard.json?ref=main`) source='courses/dase7506/data/leaderboard.json';
    else throw new Error(`Blocked unexpected network request: ${value}`);
    const content=await fs.readFile(findOfficial(source));
    fetchLog.push({requested_url:value,served_snapshot_path:source,sha256:hash(content),network_used:false});
    return new Response(content,{status:200,headers:{'Content-Type':source.endsWith('.json')?'application/json':'text/plain'}});
  };
  let issueUrl, title, body, payloadJson, payload, phaseBadge, syncStatus, artifactBinding;
  const restoreCrypto=installCryptoOverride(observer.crypto);
  try {
    // Import executes the unmodified, hashed page script and registers its own handlers.
    await import(pathToFileURL(entry).href);
    const stop=Date.now()+3000;
    while(!nodes.get('sync-status').textContent.startsWith('Updated:') && Date.now()<stop) await delay(10);
    assert(nodes.get('sync-status').textContent.startsWith('Updated:'),'Official refresh handler did not accept the captured board');
    assert(nodes.get('phase-badge').textContent==='Score submissions open','Official configureForm did not enter score phase');
    nodes.get('student-id').value=args['--student-id'];
    nodes.get('score').value=String(score.accuracy);
    nodes.get('code-url').value=args['--code-url'];
    nodes.get('checkpoint-url').value=args['--checkpoint-url'];
    nodes.get('consent').checked=true;
    nodes.get('consent').required=true;
    // The real browser would block submit on a missing required checkbox. This
    // explicit precondition preserves that HTML constraint in this DOM adapter.
    assert(nodes.get('consent').checked && !nodes.get('consent').disabled,'Result declaration not checked or disabled');
    assert(!nodes.get('student-id').disabled && !nodes.get('score').disabled,'Score fields disabled');
    assert(arena.validStudentID(nodes.get('student-id').value) && arena.validScore('mp2',Number(nodes.get('score').value)),'Official native-field equivalent validation failed');
    const handlers=nodes.get('submission-form').listeners.get('submit')||[];
    assert(handlers.length===1,'Expected exactly one official submission-form submit handler');
    let prevented=false;
    await handlers[0]({preventDefault(){prevented=true;}});
    assert(prevented,'Official form handler did not prevent default navigation');
    const statusNodes=descendants(nodes.get('submission-status'));
    const errors=statusNodes.filter(node=>node.className==='error');
    assert(errors.length===0,`Official form handler rejected preparation: ${errors.map(node=>node.textContent).join('; ')}`);
    const links=statusNodes.filter(node=>node.tagName==='A' && node.href);
    assert(links.length===1,'Official prepared() did not emit exactly one issue link');
    issueUrl=links[0].href;
    const url=new URL(issueUrl);
    assert(url.origin==='https://github.com' && url.pathname===`/${arena.REPO}/issues/new`,'Official output targets an unexpected issue repository');
    title=url.searchParams.get('title'); body=url.searchParams.get('body');
    const blocks=[...body.matchAll(/```json\s*\n([\s\S]*?)\n```/g)];
    assert(blocks.length===1,'Expected one official JSON payload in issue body');
    payloadJson=blocks[0][1]; payload=JSON.parse(payloadJson);
    assert(arena.validateSubmission(payload) && arena.validSealedArtifacts(payload.artifacts),'Official generated payload validation failed');
    assert(payload.project==='mp2' && payload.student_id===args['--student-id'] && payload.score===score.accuracy,'Generated payload differs from supplied identity or score');
    assert(arena.makeIssueURL(payload)===issueUrl,'Round trip through official makeIssueURL changed the generated issue');
    artifactBinding=observer.finish(payload.artifacts,observerProvenance);
    phaseBadge=nodes.get('phase-badge').textContent;
    syncStatus=nodes.get('sync-status').textContent;
  } finally {
    restoreCrypto();
    for(const [key,value] of Object.entries(preserved)) globalThis[key]=value;
  }
  assert(arena.submissionPhase('mp2',board.publication)==='scores','Deadline passed while preparing; no files written');
  const generatedAt=new Date().toISOString();
  const bindingBytes=Buffer.from(JSON.stringify(artifactBinding,null,2)+'\n','utf8');
  const evidence={
    status:'prepared_not_submitted', started_at_utc:startedAt, generated_at_utc:generatedAt,
    student_id:args['--student-id'], intended_github_account:account, expected_issue_author_type:'User',
    score_file:path.relative(root,scorePath).split(path.sep).join('/'),score_file_sha256:hash(scoreBytes),
    official_score:score,freeze_file_sha256:hash(freezeBytes),freeze_source_commit:freeze.source_commit,
    final_code_commit:codeCommit,code_url:args['--code-url'],checkpoint_url:args['--checkpoint-url'],
    official_source_commit:snapshot.official_site_commit,official_source_snapshot_sha256:hash(snapshotBytes),
    official_executed_entry:snapshot.published_entry_path,
    official_source_files:verified.map(({absolute_local_path,...item})=>item),
    declaration_text:declaration,declaration_checked:true,
    declaration_basis:'The user authorized formal submission of this user-owned coursework result. AI assistance is disclosed in the repository; this record does not claim a separate human understanding or manual verification exercise occurred.',
    phase:'scores',official_phase_badge:phaseBadge,official_sync_status:syncStatus,
    captured_board_generated_at:board.generated_at,captured_board_publication:board.publication,
    adapter_scope:'Minimal local DOM adapter; unchanged actually published handler and dependencies. A pass-through WebCrypto observer binds the public artifact-link plaintext and returned encryption bytes; it does not inspect RSA input key bytes, alter encryption, or retain keys. No browser-control API. window.isSecureContext is emulated only to permit local WebCrypto. All fetch calls served fixed official snapshots; no network or issue creation.',
    artifact_binding:artifactBinding,artifact_binding_file:'ARTIFACT_BINDING.json',artifact_binding_file_sha256:hash(bindingBytes),
    fetch_log:fetchLog,intervals_suppressed:timerRegistrations.map(item=>item.milliseconds),
    official_payload_json_text:payloadJson,official_payload:payload,
    issue_title:title,issue_body:body,issue_url:issueUrl,
    issue_title_sha256:hash(title),issue_body_sha256:hash(body),issue_url_sha256:hash(issueUrl),
    publication_complete:false,
  };
  // Never silently replace a prior generated encrypted payload or its evidence.
  await fs.writeFile(path.join(outputDir,'ARTIFACT_BINDING.json'),bindingBytes,{flag:'wx'});
  await fs.writeFile(path.join(outputDir,'GENERATED_FORM.json'),JSON.stringify(evidence,null,2)+'\n',{encoding:'utf8',flag:'wx'});
  await fs.writeFile(path.join(outputDir,'issue-title.txt'),title,{encoding:'utf8',flag:'wx'});
  await fs.writeFile(path.join(outputDir,'issue-body.txt'),body,{encoding:'utf8',flag:'wx'});
  await fs.writeFile(path.join(outputDir,'issue-url.txt'),issueUrl,{encoding:'utf8',flag:'wx'});
  console.log(JSON.stringify({status:evidence.status,generated_at_utc:generatedAt,official_executed_entry:evidence.official_executed_entry,phase:evidence.phase,score:score.accuracy,final_code_commit:codeCommit,output_files:outputs.map(name=>path.join(outputDir,name))},null,2));
}

main().catch(error=>{console.error(error.stack||String(error));process.exitCode=1;});
