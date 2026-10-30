# V3 official form artifact binding

The V3 local DOM adapter executes the unmodified, hash-verified published MP2
handler. That handler encrypts public artifact URLs before adding them to the
official issue payload. The payload's `artifacts` field is a sealed envelope,
not a plaintext `code_url`/`checkpoint_url` object.

`submission/v3/artifact_binding.mjs` supplies a pass-through WebCrypto observer.
It requires exactly one AES-GCM encryption of the expected public link JSON,
followed by one RSA-OAEP wrapping operation. Native arguments, keys, randomness,
algorithms and returned objects pass through unchanged. The observer does not
inspect, copy, hash, log or persist the RSA input, which contains the raw AES
key. It records public link plaintext, the IV, and hashes of returned ciphertext
and wrapped-key bytes. It checks those bytes against the envelope emitted by
the official handler. No private key is requested or retained.

This is a substantive V3 adapter change, not a path-only copy of the V1 adapter.
V1 files and `create_adapters.py` remain unchanged. The change requires review
before the final form is generated. Re-running the adapter generator must not
overwrite this reviewed V3 observer integration.

The adapter writes `ARTIFACT_BINDING.json` exclusively and stores the same
object in `GENERATED_FORM.json.artifact_binding`. The generated record also
contains `artifact_binding_file: "ARTIFACT_BINDING.json"` and
`artifact_binding_file_sha256` over the exact standalone UTF-8 bytes. Its schema
is `mp2-v3-artifact-binding-v1`, with these fields:

- `plaintext_exact`: exact official AES input JSON, ordered as `schema`,
  `project`, `submission`, `code_url`, `checkpoint_url`; schema is
  `dase7506/artifact-links-v1`, project `mp2`, submission `null`.
- `plaintext_sha256`: SHA-256 of the UTF-8 plaintext.
- `iv_sha256`, `ciphertext_sha256`, `wrapped_key_sha256`: SHA-256 of raw decoded
  envelope bytes, not the base64 strings.
- `adapter_sha256`, `observer_sha256`, `official_entry_sha256`: source hashes of
  the adapter, this observer helper, and the executed official entry module.

The submit validator must check the standalone file hash, equality with the
embedded object, exact public URLs/immutable commit against publication
verification, decoded envelope hashes, and all three source hashes. It must
also run the unchanged official payload validator and issue formatter. The
scoring protocol and form protocol are distinct and are checked separately.
The observer record is a reproducible local execution attestation; its hashes
alone do not enable a third party to decrypt the envelope.

Synthetic verification command (no official form execution, score files,
network, validation data or test data):

```powershell
node code/tests/test_artifact_binding_v3.mjs
node --check submission/v3/prepare_official_form.mjs
```

The tests exercise real native AES-256-GCM/RSA-OAEP encryption using disposable
synthetic keys, byte-for-byte decryptability, argument/result pass-through,
opaque RSA input, exact public plaintext, envelope tampering, duplicate or
missing calls, malformed base64, and restoration of the original global
crypto descriptor after failure. Synthetic keys never enter saved artifacts.

Actual result: all 11 tests passed under Node v24.14.0 (844.8646 ms reported by
the test runner), saved in
`results/research_v3/artifact-binding-synthetic-tests-direct.log`. The initial
`node --test` invocation was prevented from spawning its test subprocess by
Windows (`EPERM`), before test logic ran; that failure is retained separately
in `artifact-binding-synthetic-tests.log`. The direct invocation runs the same
`node:test` cases within one process. The adapter also passed `node --check`.

This implementation and its synthetic tests do not claim that a real V3 form
has been generated, published, or submitted.
