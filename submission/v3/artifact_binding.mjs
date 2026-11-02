/** Observe only public artifact-link plaintext and encryption outputs.
 * Native algorithms, arguments, randomness, keys, and results pass through.
 * In particular the RSA input (the raw AES key) is never read or copied here.
 */
import {createHash} from 'node:crypto';

const sha256 = value => createHash('sha256').update(value).digest('hex');
function requireThat(condition, message) { if (!condition) throw new Error(message); }
function copyBytes(value) {
  if (ArrayBuffer.isView(value)) return Buffer.from(new Uint8Array(value.buffer, value.byteOffset, value.byteLength));
  requireThat(value instanceof ArrayBuffer, 'Expected a BufferSource');
  return Buffer.from(new Uint8Array(value));
}
function decodeBase64(value, name) {
  requireThat(typeof value === 'string' && /^[A-Za-z0-9+/]+={0,2}$/.test(value), `Invalid ${name} base64`);
  const bytes = Buffer.from(value, 'base64');
  requireThat(bytes.toString('base64') === value, `Noncanonical ${name} base64`);
  return bytes;
}

export function createArtifactObserver(nativeCrypto, links) {
  requireThat(nativeCrypto?.subtle && typeof nativeCrypto.subtle.encrypt === 'function', 'Native WebCrypto is required');
  requireThat(links?.project === 'mp2' && typeof links.code_url === 'string' && typeof links.checkpoint_url === 'string', 'Expected MP2 public artifact links');
  const plaintext = JSON.stringify({schema: 'dase7506/artifact-links-v1', project: links.project,
    submission: null, code_url: links.code_url, checkpoint_url: links.checkpoint_url});
  const plaintextBytes = Buffer.from(plaintext, 'utf8');
  const nativeSubtle = nativeCrypto.subtle;
  const records = {};
  let failed = false, aesCalls = 0, rsaCalls = 0, finished = false;
  const encrypt = async (algorithm, key, data) => {
    try {
      requireThat(!finished && !failed, 'Artifact encryption observation is closed');
      const name = typeof algorithm === 'string' ? algorithm : algorithm?.name;
      if (name === 'AES-GCM') {
        aesCalls++;
        requireThat(aesCalls === 1 && rsaCalls === 0, 'Expected exactly one AES encryption before RSA wrapping');
        // This input is public link JSON, not key material. Check bytes before
        // forwarding so an unexpected encryption input cannot be persisted.
        requireThat(copyBytes(data).equals(plaintextBytes), 'Official artifact plaintext differs from the expected public links');
        const iv = copyBytes(algorithm.iv);
        requireThat(iv.length === 12, 'Official artifact IV must contain 12 bytes');
        const output = await nativeSubtle.encrypt(algorithm, key, data);
        records.aes = {iv, ciphertext: copyBytes(output)};
        return output;
      }
      requireThat(name === 'RSA-OAEP', 'Unexpected artifact encryption algorithm');
      rsaCalls++;
      requireThat(rsaCalls === 1 && records.aes, 'Expected exactly one RSA wrapping after AES encryption');
      // DO NOT inspect, log, copy, hash, or export data/key here: data is the
      // raw AES key supplied by the unchanged official handler.
      const output = await nativeSubtle.encrypt(algorithm, key, data);
      records.rsa = {wrapped: copyBytes(output)};
      return output;
    } catch (error) {
      failed = true;
      throw error;
    }
  };
  const subtle = new Proxy(nativeSubtle, {get(target, property) {
    if (property === 'encrypt') return encrypt;
    const value = Reflect.get(target, property, target);
    return typeof value === 'function' ? value.bind(target) : value;
  }});
  const crypto = new Proxy(nativeCrypto, {get(target, property) {
    if (property === 'subtle') return subtle;
    const value = Reflect.get(target, property, target);
    return typeof value === 'function' ? value.bind(target) : value;
  }});
  return {crypto, finish(envelope, provenance) {
    requireThat(!finished && !failed && aesCalls === 1 && rsaCalls === 1 && records.aes && records.rsa,
      'Exactly one successful AES encryption and one RSA wrapping are required');
    requireThat(envelope && Object.keys(envelope).sort().join(',') === 'algorithm,ciphertext,iv,key_id,wrapped_key', 'Unexpected official sealed-artifact fields');
    requireThat(envelope.algorithm === 'RSA-OAEP-256+AES-256-GCM' && envelope.key_id === '7506-2026-v1', 'Unexpected official sealed-artifact algorithm/key ID');
    const iv = decodeBase64(envelope.iv, 'IV'), ciphertext = decodeBase64(envelope.ciphertext, 'ciphertext');
    const wrapped = decodeBase64(envelope.wrapped_key, 'wrapped key');
    requireThat(iv.equals(records.aes.iv) && ciphertext.equals(records.aes.ciphertext) && wrapped.equals(records.rsa.wrapped),
      'Official sealed-artifact bytes differ from observed encryption outputs');
    requireThat(iv.length === 12 && wrapped.length === 384 && ciphertext.length === plaintextBytes.length + 16,
      'Unexpected official artifact encryption output size');
    for (const name of ['adapter_sha256', 'observer_sha256', 'official_entry_sha256']) {
      requireThat(/^[0-9a-f]{64}$/.test(provenance?.[name]), `Missing ${name}`);
    }
    finished = true;
    return {schema: 'mp2-v3-artifact-binding-v1', plaintext_exact: plaintext,
      plaintext_sha256: sha256(plaintextBytes), iv_sha256: sha256(iv), ciphertext_sha256: sha256(ciphertext),
      wrapped_key_sha256: sha256(wrapped), adapter_sha256: provenance.adapter_sha256,
      observer_sha256: provenance.observer_sha256, official_entry_sha256: provenance.official_entry_sha256};
  }};
}

/** Preserve the exact original descriptor, including Node's getter if present. */
export function installCryptoOverride(crypto, target = globalThis) {
  const descriptor = Object.getOwnPropertyDescriptor(target, 'crypto');
  if (descriptor && !descriptor.configurable) {
    requireThat(Object.hasOwn(descriptor, 'value') && descriptor.writable, 'Global crypto cannot be temporarily overridden');
    Object.defineProperty(target, 'crypto', {...descriptor, value: crypto});
  } else {
    Object.defineProperty(target, 'crypto', {value: crypto, writable: true, configurable: true,
      enumerable: descriptor?.enumerable ?? true});
  }
  let restored = false;
  return () => {
    if (restored) return;
    if (descriptor) Object.defineProperty(target, 'crypto', descriptor);
    else delete target.crypto;
    restored = true;
  };
}
