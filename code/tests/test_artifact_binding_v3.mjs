import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createHash, webcrypto} from 'node:crypto';
import {createArtifactObserver, installCryptoOverride} from '../../submission/v3/artifact_binding.mjs';

const links = {project:'mp2',code_url:'https://github.com/example/repo/tree/'+'a'.repeat(40),
  checkpoint_url:'https://github.com/example/repo/tree/'+'a'.repeat(40)+'/artifacts/research_v3/final_selected'};
const publicText = JSON.stringify({schema:'dase7506/artifact-links-v1',project:'mp2',submission:null,
  code_url:links.code_url,checkpoint_url:links.checkpoint_url});
const sha = value => createHash('sha256').update(value).digest('hex');
const provenance = {adapter_sha256:'a'.repeat(64),observer_sha256:'b'.repeat(64),official_entry_sha256:'c'.repeat(64)};
const base64 = value => Buffer.from(value).toString('base64');
const envelope = (iv,ciphertext,wrapped) => ({algorithm:'RSA-OAEP-256+AES-256-GCM',key_id:'7506-2026-v1',
  iv:base64(iv),ciphertext:base64(ciphertext),wrapped_key:base64(wrapped)});

function fakeNative() {
  const calls=[];
  const aesOutput=new Uint8Array(Buffer.byteLength(publicText)+16).fill(11).buffer;
  const rsaOutput=new Uint8Array(384).fill(22).buffer;
  const native={subtle:{async encrypt(algorithm,key,data) {
    assert.equal(this,native.subtle);
    calls.push([algorithm,key,data]);
    return algorithm.name==='AES-GCM' ? aesOutput : rsaOutput;
  }, passthrough(){assert.equal(this,native.subtle);return 10;}},
  getRandomValues(value){assert.equal(this,native);return value.fill(3);}};
  return {native,calls,aesOutput,rsaOutput};
}
async function fakeEncryption(options={}) {
  const setup=fakeNative(), observer=createArtifactObserver(setup.native,links);
  const iv=new Uint8Array(12).fill(7), key={synthetic:true};
  const aesAlgorithm={name:'AES-GCM',iv};
  const aesData=Buffer.from(publicText);
  const ciphertext=await observer.crypto.subtle.encrypt(aesAlgorithm,key,aesData);
  // An observer that reads *any* property of RSA data fails this test. Native
  // fake transport only stores the object reference, as an opaque argument.
  const opaqueRsaData=new Proxy({}, {get(){throw new Error('RSA key input was inspected');}});
  const rsaAlgorithm={name:'RSA-OAEP'};
  const wrapped=options.omitRsa ? undefined : await observer.crypto.subtle.encrypt(rsaAlgorithm,key,opaqueRsaData);
  return {...setup,observer,iv,key,aesAlgorithm,aesData,opaqueRsaData,rsaAlgorithm,ciphertext,wrapped,
    payload:envelope(iv,ciphertext,wrapped || setup.rsaOutput)};
}

test('real native AES-GCM/RSA-OAEP encrypt/decrypt and exact public binding',async()=>{
  const observer=createArtifactObserver(webcrypto,links), crypto=observer.crypto;
  const key=await crypto.subtle.generateKey({name:'AES-GCM',length:256},true,['encrypt','decrypt']);
  const pair=await crypto.subtle.generateKey({name:'RSA-OAEP',modulusLength:3072,
    publicExponent:new Uint8Array([1,0,1]),hash:'SHA-256'},true,['encrypt','decrypt']);
  const iv=crypto.getRandomValues(new Uint8Array(12));
  const ciphertext=await crypto.subtle.encrypt({name:'AES-GCM',iv},key,Buffer.from(publicText));
  const rawKey=await crypto.subtle.exportKey('raw',key);
  const wrapped=await crypto.subtle.encrypt({name:'RSA-OAEP'},pair.publicKey,rawKey);
  const record=observer.finish(envelope(iv,ciphertext,wrapped),provenance);
  assert.equal(record.plaintext_exact,publicText);
  assert.equal(record.plaintext_sha256,sha(Buffer.from(publicText)));
  assert.equal(record.iv_sha256,sha(iv));
  assert.equal(record.ciphertext_sha256,sha(Buffer.from(ciphertext)));
  assert.equal(record.wrapped_key_sha256,sha(Buffer.from(wrapped)));
  assert.deepEqual(Object.keys(record).sort(),['schema','plaintext_exact','plaintext_sha256','iv_sha256',
    'ciphertext_sha256','wrapped_key_sha256','adapter_sha256','observer_sha256','official_entry_sha256'].sort());
  const unwrapped=await webcrypto.subtle.decrypt({name:'RSA-OAEP'},pair.privateKey,wrapped);
  assert.deepEqual(Buffer.from(unwrapped),Buffer.from(rawKey));
  const decoded=await webcrypto.subtle.decrypt({name:'AES-GCM',iv},key,ciphertext);
  assert.equal(Buffer.from(decoded).toString('utf8'),publicText);
  assert.equal(JSON.stringify(record).includes(base64(rawKey)),false);
});

test('all native arguments and output identities pass through; RSA input is opaque',async()=>{
  const x=await fakeEncryption();
  assert.equal(x.ciphertext,x.aesOutput); assert.equal(x.wrapped,x.rsaOutput);
  assert.deepEqual(x.calls[0],[x.aesAlgorithm,x.key,x.aesData]);
  assert.equal(x.calls[1][0],x.rsaAlgorithm); assert.equal(x.calls[1][1],x.key);
  assert.equal(x.calls[1][2],x.opaqueRsaData);
  assert.equal(x.observer.crypto.subtle.passthrough(),10);
  assert.deepEqual(x.observer.crypto.getRandomValues(new Uint8Array(2)),new Uint8Array([3,3]));
  x.observer.finish(x.payload,provenance);
});

test('wrong public plaintext is rejected before native encryption and cannot finish',async()=>{
  const x=fakeNative(), observer=createArtifactObserver(x.native,links);
  await assert.rejects(observer.crypto.subtle.encrypt({name:'AES-GCM',iv:new Uint8Array(12)}, {}, Buffer.from('{}')),/plaintext differs/);
  assert.equal(x.calls.length,0);
  assert.throws(()=>observer.finish({},provenance),/successful AES/);
});

test('ArrayBuffer view offset is respected for plaintext and IV',async()=>{
  const x=fakeNative(), observer=createArtifactObserver(x.native,links);
  const data=Buffer.concat([Buffer.from('prefix'),Buffer.from(publicText),Buffer.from('suffix')]);
  const ivStorage=new Uint8Array(30), iv=ivStorage.subarray(9,21);
  const output=await observer.crypto.subtle.encrypt({name:'AES-GCM',iv},{},data.subarray(6,data.length-6));
  const wrapped=await observer.crypto.subtle.encrypt({name:'RSA-OAEP'},{},{});
  assert.equal(observer.finish(envelope(iv,output,wrapped),provenance).iv_sha256,sha(iv));
});

test('changed ciphertext, IV or wrapped output is rejected',async()=>{
  for(const field of ['ciphertext','iv','wrapped_key']) {
    const x=await fakeEncryption(), changed={...x.payload};
    const bytes=Buffer.from(changed[field],'base64');bytes[0]^=1;changed[field]=bytes.toString('base64');
    assert.throws(()=>x.observer.finish(changed,provenance),/differ from observed/);
  }
});

test('missing RSA wrapping and extra AES/RSA calls are rejected',async()=>{
  const missing=await fakeEncryption({omitRsa:true});
  assert.throws(()=>missing.observer.finish(missing.payload,provenance),/successful AES/);
  for(const name of ['AES-GCM','RSA-OAEP']) {
    const x=await fakeEncryption();
    await assert.rejects(x.observer.crypto.subtle.encrypt({name,iv:x.iv},x.key,x.aesData),/exactly one/);
    assert.throws(()=>x.observer.finish(x.payload,provenance),/successful AES/);
  }
});

test('RSA before AES and unexpected encryption algorithms fail closed',async()=>{
  for(const name of ['RSA-OAEP','AES-CBC']) {
    const x=fakeNative(), observer=createArtifactObserver(x.native,links);
    await assert.rejects(observer.crypto.subtle.encrypt({name},{},{}));
    assert.equal(x.calls.length,0);
  }
});

test('noncanonical base64, extra fields, wrong key ID and missing source hash rejected',async()=>{
  const changes=[x=>{x.iv+='=';},x=>{x.extra=true;},x=>{x.key_id='other';}];
  for(const change of changes) {
    const x=await fakeEncryption();change(x.payload);
    assert.throws(()=>x.observer.finish(x.payload,provenance));
  }
  const x=await fakeEncryption();
  assert.throws(()=>x.observer.finish(x.payload,{...provenance,observer_sha256:undefined}),/observer_sha256/);
});

test('native encryption errors are preserved and make observation unusable',async()=>{
  const native=fakeNative().native, original=new Error('synthetic native error');
  native.subtle.encrypt=async()=>{throw original;};
  const observer=createArtifactObserver(native,links);
  await assert.rejects(observer.crypto.subtle.encrypt({name:'AES-GCM',iv:new Uint8Array(12)},{},Buffer.from(publicText)),e=>e===original);
  assert.throws(()=>observer.finish({},provenance),/successful AES/);
});

test('finished observation cannot encrypt again or generate a second binding',async()=>{
  const x=await fakeEncryption();x.observer.finish(x.payload,provenance);
  assert.throws(()=>x.observer.finish(x.payload,provenance),/successful AES/);
  await assert.rejects(x.observer.crypto.subtle.encrypt({name:'RSA-OAEP'},{},{}),/closed/);
});

test('temporary global override restores original descriptor after exception',()=>{
  const original=Object.getOwnPropertyDescriptor(globalThis,'crypto'), marker={subtle:{}};
  const restore=installCryptoOverride(marker);
  try {assert.equal(globalThis.crypto,marker);throw new Error('synthetic handler failure');}
  catch(error){assert.equal(error.message,'synthetic handler failure');}
  finally {restore();restore();}
  assert.deepEqual(Object.getOwnPropertyDescriptor(globalThis,'crypto'),original);
  const target={};const undo=installCryptoOverride(marker,target);undo();
  assert.equal(Object.hasOwn(target,'crypto'),false);
});
