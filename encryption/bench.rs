#![doc = r#"
# StringWars: Encryption Benchmarks

This file contains benchmarks for various Rust encryption libraries, comparing AEAD (Authenticated Encryption with
Associated Data) ciphers commonly used in TLS 1.2/1.3 and the Noise Protocol Framework.

The benchmarks focus on:
- **AES-256-GCM**: Hardware-accelerated AEAD cipher
- **ChaCha20-Poly1305**: Software-optimized AEAD cipher

The benchmarks are organized into three categories:

**Key Generation/Setup**:
- Ring key generation
- OpenSSL cipher initialization

**Encryption** (encrypting dataset tokens):
- ChaCha20-Poly1305 via Ring
- AES-256-GCM via Ring
- ChaCha20-Poly1305 IETF via OpenSSL
- AES-256-GCM via OpenSSL
- AES-256-GCM via StringZilla
- AES-256-CTR via StringZilla (unauthenticated, so not comparable on security)
- ChaCha20-Poly1305 IETF via libsodium
- XChaCha20-Poly1305 IETF via libsodium (extended nonce)

**Decryption** (decrypting previously encrypted data):
- ChaCha20-Poly1305 via Ring
- AES-256-GCM via Ring
- ChaCha20-Poly1305 IETF via OpenSSL
- AES-256-GCM via OpenSSL
- AES-256-GCM via StringZilla
- AES-256-CTR via StringZilla (its own inverse, so the same call as encryption)
- ChaCha20-Poly1305 IETF via libsodium
- XChaCha20-Poly1305 IETF via libsodium (extended nonce)

## System Dependencies

Before running these benchmarks, ensure the following system packages are installed:

```sh
sudo apt install -y build-essential pkg-config libssl-dev libsodium-dev # for Ubuntu/Debian
sudo dnf install -y gcc pkg-config openssl-devel libsodium-devel # for RHEL/Fedora
brew install pkg-config openssl libsodium # for macOS
```

## Usage Examples

The benchmarks use environment variables to control the input dataset:

- `STRINGWARS_DATASET`: Path to the input dataset file (required)
- `STRINGWARS_TOKENS`: Specifies how to interpret the input. Allowed values:
  - `lines`: Process the dataset line by line
  - `words`: Process the dataset word by word
  - `file`: Process the entire file as a single token
- `STRINGWARS_FILTER`: Regex pattern to filter which benchmarks to run (e.g., `aes` for AES benchmarks,
  `encryption/.*chacha` for ChaCha)

To run the benchmarks with the appropriate CPU features enabled:

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_TOKENS=lines \
    cargo bench --features bench_encryption --bench bench_encryption
```

To benchmark Ring vs OpenSSL encryption and decryption:

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=data/acgt/acgt_100.txt \
    STRINGWARS_TOKENS=lines \
    STRINGWARS_FILTER="(cryption/openssl|cryption/ring)" \
    cargo bench --features bench_encryption --bench bench_encryption
```
"#]
use std::hint::black_box;

use openssl::symm::{Cipher, Crypter, Mode};
use stringtape::BytesCowsAuto;
use stringzilla::sz::{Aes256CtrKey, Aes256GcmKey, AES256_NONCE_LENGTH, AES256_TAG_LENGTH};

#[path = "../utils.rs"]
mod utils;
use utils::{
    install_panic_hook, load_dataset, log_stringzilla_metadata, measure_throughput, BenchBudget,
    ReportAs, ResultExt, WorkUnits,
};

/// Constructs the Ring AEAD nonce belonging to one token, from its index in the dataset.
/// The index occupies the first 8 bytes of the 12-byte nonce in little-endian order.
///
/// Every backend derives its nonce from the token index, so a decryption call reconstructs the
/// nonce its token was sealed under however many times the timed loop has cycled the dataset.
fn ring_nonce_for_index(index: usize) -> ring::aead::Nonce {
    let mut nonce_bytes = [0u8; ring::aead::NONCE_LEN];
    nonce_bytes[..8].copy_from_slice(&(index as u64).to_le_bytes());
    ring::aead::Nonce::try_assume_unique_for_key(&nonce_bytes).unwrap()
}

/// Constructs the OpenSSL 12-byte IV belonging to one token, from its index in the dataset.
fn openssl_iv_for_index(index: usize) -> [u8; 12] {
    let mut iv = [0u8; 12];
    iv[..8].copy_from_slice(&(index as u64).to_le_bytes());
    iv
}

/// Constructs the StringZilla 12-byte nonce belonging to one token, from its index in the dataset.
fn stringzilla_nonce_for_index(index: usize) -> [u8; AES256_NONCE_LENGTH] {
    let mut nonce = [0u8; AES256_NONCE_LENGTH];
    nonce[..8].copy_from_slice(&(index as u64).to_le_bytes());
    nonce
}

/// Constructs the libsodium ChaCha20-Poly1305 IETF nonce belonging to one token, from its index.
fn sodium_chacha20_nonce_for_index(
    index: usize,
) -> sodiumoxide::crypto::aead::chacha20poly1305_ietf::Nonce {
    use sodiumoxide::crypto::aead::chacha20poly1305_ietf;
    let mut nonce_bytes = [0u8; chacha20poly1305_ietf::NONCEBYTES];
    nonce_bytes[..8].copy_from_slice(&(index as u64).to_le_bytes());
    chacha20poly1305_ietf::Nonce(nonce_bytes)
}

/// Constructs the libsodium XChaCha20-Poly1305 IETF nonce belonging to one token, from its index.
fn sodium_xchacha20_nonce_for_index(
    index: usize,
) -> sodiumoxide::crypto::aead::xchacha20poly1305_ietf::Nonce {
    use sodiumoxide::crypto::aead::xchacha20poly1305_ietf;
    let mut nonce_bytes = [0u8; xchacha20poly1305_ietf::NONCEBYTES];
    nonce_bytes[..8].copy_from_slice(&(index as u64).to_le_bytes());
    xchacha20poly1305_ietf::Nonce(nonce_bytes)
}

/// Benchmarks key generation and cipher setup overhead. Each variant builds one key/cipher per
/// call and cycles for the budget; throughput is reported as bytes/s over the 32-byte key.
fn bench_key_generation(budget: &BenchBudget) {
    use ring::aead;

    // Benchmark: ring ChaCha20-Poly1305 key generation
    measure_throughput(
        "keygen/ring::chacha20poly1305",
        ReportAs::Bytes,
        budget,
        || {
            let key_bytes = [0u8; 32]; // 256-bit key
            let key = aead::UnboundKey::new(&aead::CHACHA20_POLY1305, &key_bytes);
            let _ = black_box(key);
            WorkUnits::bytes(32)
        },
    );

    // Benchmark: ring AES-256-GCM key generation
    measure_throughput("keygen/ring::aes256gcm", ReportAs::Bytes, budget, || {
        let key_bytes = [0u8; 32]; // 256-bit key
        let key = aead::UnboundKey::new(&aead::AES_256_GCM, &key_bytes);
        let _ = black_box(key);
        WorkUnits::bytes(32)
    });

    // Benchmark: OpenSSL ChaCha20-Poly1305 initialization
    {
        measure_throughput(
            "keygen/openssl::chacha20poly1305",
            ReportAs::Bytes,
            budget,
            || {
                let key = [0u8; 32];
                let cipher = Cipher::chacha20_poly1305();
                // OpenSSL refuses to build a context for either cipher without a nonce, so setup is
                // measured with the same 12-byte one the encryption rows use.
                let crypter = Crypter::new(cipher, Mode::Encrypt, &key, Some(&[0u8; 12]));
                let _ = black_box(crypter);
                WorkUnits::bytes(32)
            },
        );
    }

    // Benchmark: OpenSSL AES-256-GCM initialization
    {
        measure_throughput("keygen/openssl::aes256gcm", ReportAs::Bytes, budget, || {
            let key = [0u8; 32];
            let cipher = Cipher::aes_256_gcm();
            let crypter = Crypter::new(cipher, Mode::Encrypt, &key, Some(&[0u8; 12]));
            let _ = black_box(crypter);
            WorkUnits::bytes(32)
        });
    }

    // Benchmark: StringZilla AES-256-GCM key expansion, schedule plus the eight hash subkey powers
    measure_throughput(
        "keygen/stringzilla::aes256gcm",
        ReportAs::Bytes,
        budget,
        || {
            let key = Aes256GcmKey::new(&[0u8; 32]);
            let _ = black_box(&key);
            WorkUnits::bytes(32)
        },
    );

    // Benchmark: StringZilla AES-256-CTR key expansion. Counter mode reads no hash subkey powers, so
    // its key is the bare round-key schedule and expanding it should cost strictly less than the above.
    measure_throughput(
        "keygen/stringzilla::aes256ctr",
        ReportAs::Bytes,
        budget,
        || {
            let key = Aes256CtrKey::new(&[0u8; 32]);
            let _ = black_box(&key);
            WorkUnits::bytes(32)
        },
    );

    // Benchmark: libsodium ChaCha20-Poly1305 IETF key generation
    {
        use sodiumoxide::crypto::aead::chacha20poly1305_ietf::{self, Key};
        measure_throughput(
            "keygen/libsodium::chacha20poly1305_ietf",
            ReportAs::Bytes,
            budget,
            || {
                let key = Key([0u8; chacha20poly1305_ietf::KEYBYTES]);
                let _ = black_box(key);
                WorkUnits::bytes(chacha20poly1305_ietf::KEYBYTES as u64)
            },
        );
    }

    // Benchmark: libsodium XChaCha20-Poly1305 IETF key generation
    {
        use sodiumoxide::crypto::aead::xchacha20poly1305_ietf::{self, Key};
        measure_throughput(
            "keygen/libsodium::xchacha20poly1305_ietf",
            ReportAs::Bytes,
            budget,
            || {
                let key = Key([0u8; xchacha20poly1305_ietf::KEYBYTES]);
                let _ = black_box(key);
                WorkUnits::bytes(xchacha20poly1305_ietf::KEYBYTES as u64)
            },
        );
    }
}

/// Seals one message with OpenSSL, writing the ciphertext into `sealed` and the tag into `tag`.
/// A context is built per message because the crate exposes no way to re-key an existing one.
fn seal_with_openssl(
    cipher: Cipher,
    iv: &[u8],
    plaintext: &[u8],
    sealed: &mut [u8],
    tag: &mut [u8],
) {
    let mut crypter = Crypter::new(cipher, Mode::Encrypt, &[0u8; 32], Some(iv)).unwrap();
    let written = crypter.update(plaintext, sealed).unwrap();
    let _ = crypter.finalize(&mut sealed[written..]).unwrap();
    crypter.get_tag(tag).unwrap();
}

/// Times one cipher over the dataset, cycling tokens for the budget and handing `cipher` the index
/// its nonce derives from alongside the token itself. Throughput is reported as bytes/s over the
/// plaintext, so encryption and decryption rows account for the same bytes.
fn measure_per_token(
    name: &str,
    budget: &BenchBudget,
    tokens: &[&[u8]],
    mut cipher: impl FnMut(usize, &[u8]),
) {
    let mut cursor = 0usize;
    measure_throughput(name, ReportAs::Bytes, budget, || {
        let index = cursor % tokens.len();
        let token = tokens[index];
        cursor += 1;
        cipher(index, token);
        WorkUnits::new(1, token.len() as u64)
    });
}

/// Benchmarks AEAD encryption (encrypt + authenticate). Each variant encrypts one token per call
/// and cycles the dataset for the budget; throughput is reported as bytes/s over the plaintext.
fn bench_encryption(budget: &BenchBudget, tokens: &BytesCowsAuto) {
    use ring::aead::{self, Aad, LessSafeKey, UnboundKey};

    // Collect token slices once so each cyclic call indexes a single token.
    let slices: Vec<&[u8]> = tokens.iter().collect();

    // Widest token any row has to hold, so each sizes its scratch buffer outside the timed loop and
    // reports the cipher rather than the allocator.
    let longest_token = slices.iter().map(|token| token.len()).max().unwrap_or(0);

    // Benchmark: ring ChaCha20-Poly1305 encryption
    {
        let key = LessSafeKey::new(UnboundKey::new(&aead::CHACHA20_POLY1305, &[0u8; 32]).unwrap());
        let mut sealed = Vec::with_capacity(longest_token + aead::CHACHA20_POLY1305.tag_len());
        measure_per_token(
            "encryption/ring::chacha20poly1305",
            budget,
            &slices,
            |index, token| {
                sealed.clear();
                sealed.extend_from_slice(token);
                let nonce = ring_nonce_for_index(index);
                let _ = black_box(key.seal_in_place_append_tag(nonce, Aad::empty(), &mut sealed));
            },
        );
    }

    // Benchmark: ring AES-256-GCM encryption
    {
        let key = LessSafeKey::new(UnboundKey::new(&aead::AES_256_GCM, &[0u8; 32]).unwrap());
        let mut sealed = Vec::with_capacity(longest_token + aead::AES_256_GCM.tag_len());
        measure_per_token(
            "encryption/ring::aes256gcm",
            budget,
            &slices,
            |index, token| {
                sealed.clear();
                sealed.extend_from_slice(token);
                let nonce = ring_nonce_for_index(index);
                let _ = black_box(key.seal_in_place_append_tag(nonce, Aad::empty(), &mut sealed));
            },
        );
    }

    // Benchmark: OpenSSL ChaCha20-Poly1305 encryption. The cipher context is per message, because
    // the crate exposes no way to re-key an existing one with a new nonce.
    {
        let cipher = Cipher::chacha20_poly1305();
        let mut sealed = vec![0u8; longest_token + cipher.block_size()];
        let mut tag = [0u8; AES256_TAG_LENGTH];
        measure_per_token(
            "encryption/openssl::chacha20poly1305",
            budget,
            &slices,
            |index, token| {
                seal_with_openssl(
                    cipher,
                    &openssl_iv_for_index(index),
                    token,
                    &mut sealed,
                    &mut tag,
                );
                let _ = black_box(&tag);
            },
        );
    }

    // Benchmark: OpenSSL AES-256-GCM encryption
    {
        let cipher = Cipher::aes_256_gcm();
        let mut sealed = vec![0u8; longest_token + cipher.block_size()];
        let mut tag = [0u8; AES256_TAG_LENGTH];
        measure_per_token(
            "encryption/openssl::aes256gcm",
            budget,
            &slices,
            |index, token| {
                seal_with_openssl(
                    cipher,
                    &openssl_iv_for_index(index),
                    token,
                    &mut sealed,
                    &mut tag,
                );
                let _ = black_box(&tag);
            },
        );
    }

    // Benchmark: StringZilla AES-256-GCM encryption
    {
        let key = Aes256GcmKey::new(&[0u8; 32]);
        let mut sealed = vec![0u8; longest_token];
        measure_per_token(
            "encryption/stringzilla::aes256gcm",
            budget,
            &slices,
            |index, token| {
                let nonce = stringzilla_nonce_for_index(index);
                let tag = key.encrypt_into(&nonce, &[], token, &mut sealed[..token.len()]);
                let _ = black_box(tag);
            },
        );
    }

    // Benchmark: StringZilla AES-256-CTR encryption. Unauthenticated, so it is not comparable to the
    // AEAD rows above on security, only on what the cipher itself costs without a tag to accumulate.
    {
        let key = Aes256CtrKey::new(&[0u8; 32]);
        let mut sealed = vec![0u8; longest_token];
        measure_per_token(
            "encryption/stringzilla::aes256ctr",
            budget,
            &slices,
            |index, token| {
                let nonce = stringzilla_nonce_for_index(index);
                key.xor_into(&nonce, 0, token, &mut sealed[..token.len()]);
                let _ = black_box(&sealed[..token.len()]);
            },
        );
    }

    // Benchmark: libsodium ChaCha20-Poly1305 IETF encryption
    {
        use sodiumoxide::crypto::aead::chacha20poly1305_ietf::{self, Key};

        let key = Key([0u8; chacha20poly1305_ietf::KEYBYTES]);
        let mut sealed = vec![0u8; longest_token];
        measure_per_token(
            "encryption/libsodium::chacha20poly1305_ietf",
            budget,
            &slices,
            |index, token| {
                let message = &mut sealed[..token.len()];
                message.copy_from_slice(token);
                let nonce = sodium_chacha20_nonce_for_index(index);
                let tag = chacha20poly1305_ietf::seal_detached(message, None, &nonce, &key);
                let _ = black_box(tag);
            },
        );
    }

    // Benchmark: libsodium XChaCha20-Poly1305 IETF encryption
    {
        use sodiumoxide::crypto::aead::xchacha20poly1305_ietf::{self, Key};

        let key = Key([0u8; xchacha20poly1305_ietf::KEYBYTES]);
        let mut sealed = vec![0u8; longest_token];
        measure_per_token(
            "encryption/libsodium::xchacha20poly1305_ietf",
            budget,
            &slices,
            |index, token| {
                let message = &mut sealed[..token.len()];
                message.copy_from_slice(token);
                let nonce = sodium_xchacha20_nonce_for_index(index);
                let tag = xchacha20poly1305_ietf::seal_detached(message, None, &nonce, &key);
                let _ = black_box(tag);
            },
        );
    }
}

/// One backend's sealed copy of the corpus, prepared before the timed loop. The tag rides beside
/// its ciphertext rather than appended to it, so a row can open in place with the detached calls.
struct SealedCorpus<TagType> {
    ciphertexts: Vec<Vec<u8>>,
    tags: Vec<TagType>,
}

impl<TagType> SealedCorpus<TagType> {
    /// Seals every token, letting `seal` write the ciphertext into the buffer it is handed.
    fn new(tokens: &BytesCowsAuto, seal: impl Fn(usize, &[u8], &mut Vec<u8>) -> TagType) -> Self {
        let mut ciphertexts = Vec::with_capacity(tokens.len());
        let mut tags = Vec::with_capacity(tokens.len());
        for (index, token) in tokens.iter().enumerate() {
            let mut ciphertext = Vec::with_capacity(token.len() + AES256_TAG_LENGTH);
            tags.push(seal(index, token, &mut ciphertext));
            ciphertexts.push(ciphertext);
        }
        Self { ciphertexts, tags }
    }

    /// The longest sealed message, so a row sizes its scratch buffer once.
    fn longest(&self) -> usize {
        self.ciphertexts
            .iter()
            .map(|ciphertext| ciphertext.len())
            .max()
            .unwrap_or(0)
    }
}

/// Times one cipher over a sealed corpus, cycling ciphertexts for the budget. Throughput counts the
/// original plaintext lengths, so decryption rows account for the same bytes the encryption rows do.
fn measure_per_ciphertext<TagType>(
    name: &str,
    budget: &BenchBudget,
    corpus: &SealedCorpus<TagType>,
    plaintext_lengths: &[u64],
    mut cipher: impl FnMut(usize, &[u8], &TagType),
) {
    let mut cursor = 0usize;
    measure_throughput(name, ReportAs::Bytes, budget, || {
        let index = cursor % corpus.ciphertexts.len();
        cursor += 1;
        cipher(index, &corpus.ciphertexts[index], &corpus.tags[index]);
        WorkUnits::new(1, plaintext_lengths[index])
    });
}

/// Benchmarks AEAD decryption (verify + decrypt). Pre-encrypts every token once before the timed
/// loop; each measured call then decrypts one ciphertext and cycles the dataset for the budget.
/// Throughput is reported over the original plaintext lengths to match the encryption accounting.
fn bench_decryption(budget: &BenchBudget, tokens: &BytesCowsAuto) {
    use ring::aead::{self, Aad, LessSafeKey, UnboundKey};
    use sodiumoxide::crypto::aead::chacha20poly1305_ietf::{self, Key as SodiumChaCha20Key};
    use sodiumoxide::crypto::aead::xchacha20poly1305_ietf::{self, Key as SodiumXChaCha20Key};

    // Sealed buffers carry tag bytes the throughput accounting excludes, so the work per call comes
    // from the plaintext the token started as.
    let plaintext_lengths: Vec<u64> = tokens.iter().map(|token| token.len() as u64).collect();
    let key_bytes = [0u8; 32];

    // Ring appends its tag to the ciphertext rather than detaching it, so its corpus carries the
    // unit tag and each row opens the buffer whole.
    let key_ring_chacha =
        LessSafeKey::new(UnboundKey::new(&aead::CHACHA20_POLY1305, &key_bytes).unwrap());
    let ring_chacha = SealedCorpus::new(tokens, |index, token, ciphertext| {
        ciphertext.extend_from_slice(token);
        key_ring_chacha
            .seal_in_place_append_tag(ring_nonce_for_index(index), Aad::empty(), ciphertext)
            .unwrap()
    });
    let key_ring_aes = LessSafeKey::new(UnboundKey::new(&aead::AES_256_GCM, &key_bytes).unwrap());
    let ring_aes = SealedCorpus::new(tokens, |index, token, ciphertext| {
        ciphertext.extend_from_slice(token);
        key_ring_aes
            .seal_in_place_append_tag(ring_nonce_for_index(index), Aad::empty(), ciphertext)
            .unwrap()
    });

    // Benchmark: ring ChaCha20-Poly1305 decryption
    {
        let mut opened = Vec::with_capacity(ring_chacha.longest());
        measure_per_ciphertext(
            "decryption/ring::chacha20poly1305",
            budget,
            &ring_chacha,
            &plaintext_lengths,
            |index, ciphertext, _| {
                // Opening consumes the buffer, so each call works on a copy it reuses.
                opened.clear();
                opened.extend_from_slice(ciphertext);
                let nonce = ring_nonce_for_index(index);
                let _ = black_box(key_ring_chacha.open_in_place(nonce, Aad::empty(), &mut opened));
            },
        );
    }

    // Benchmark: ring AES-256-GCM decryption
    {
        let mut opened = Vec::with_capacity(ring_aes.longest());
        measure_per_ciphertext(
            "decryption/ring::aes256gcm",
            budget,
            &ring_aes,
            &plaintext_lengths,
            |index, ciphertext, _| {
                opened.clear();
                opened.extend_from_slice(ciphertext);
                let nonce = ring_nonce_for_index(index);
                let _ = black_box(key_ring_aes.open_in_place(nonce, Aad::empty(), &mut opened));
            },
        );
    }

    let cipher_chacha = Cipher::chacha20_poly1305();
    let openssl_chacha = SealedCorpus::new(tokens, |index, token, ciphertext| {
        let mut tag = [0u8; AES256_TAG_LENGTH];
        ciphertext.resize(token.len() + cipher_chacha.block_size(), 0);
        seal_with_openssl(
            cipher_chacha,
            &openssl_iv_for_index(index),
            token,
            ciphertext,
            &mut tag,
        );
        ciphertext.truncate(token.len());
        tag
    });
    let cipher_aes = Cipher::aes_256_gcm();
    let openssl_aes = SealedCorpus::new(tokens, |index, token, ciphertext| {
        let mut tag = [0u8; AES256_TAG_LENGTH];
        ciphertext.resize(token.len() + cipher_aes.block_size(), 0);
        seal_with_openssl(
            cipher_aes,
            &openssl_iv_for_index(index),
            token,
            ciphertext,
            &mut tag,
        );
        ciphertext.truncate(token.len());
        tag
    });

    // Benchmark: OpenSSL ChaCha20-Poly1305 decryption
    {
        let mut opened = vec![0u8; openssl_chacha.longest() + cipher_chacha.block_size()];
        measure_per_ciphertext(
            "decryption/openssl::chacha20poly1305",
            budget,
            &openssl_chacha,
            &plaintext_lengths,
            |index, ciphertext, tag| {
                let iv = openssl_iv_for_index(index);
                let mut crypter =
                    Crypter::new(cipher_chacha, Mode::Decrypt, &key_bytes, Some(&iv)).unwrap();
                crypter.set_tag(tag).unwrap();
                let written = crypter.update(ciphertext, &mut opened).unwrap();
                let _ = black_box(crypter.finalize(&mut opened[written..]));
            },
        );
    }

    // Benchmark: OpenSSL AES-256-GCM decryption
    {
        let mut opened = vec![0u8; openssl_aes.longest() + cipher_aes.block_size()];
        measure_per_ciphertext(
            "decryption/openssl::aes256gcm",
            budget,
            &openssl_aes,
            &plaintext_lengths,
            |index, ciphertext, tag| {
                let iv = openssl_iv_for_index(index);
                let mut crypter =
                    Crypter::new(cipher_aes, Mode::Decrypt, &key_bytes, Some(&iv)).unwrap();
                crypter.set_tag(tag).unwrap();
                let written = crypter.update(ciphertext, &mut opened).unwrap();
                let _ = black_box(crypter.finalize(&mut opened[written..]));
            },
        );
    }

    let key_stringzilla_gcm = Aes256GcmKey::new(&key_bytes);
    let stringzilla_gcm = SealedCorpus::new(tokens, |index, token, ciphertext| {
        ciphertext.resize(token.len(), 0);
        key_stringzilla_gcm.encrypt_into(
            &stringzilla_nonce_for_index(index),
            &[],
            token,
            ciphertext,
        )
    });

    // Benchmark: StringZilla AES-256-GCM decryption
    {
        let mut opened = vec![0u8; stringzilla_gcm.longest()];
        measure_per_ciphertext(
            "decryption/stringzilla::aes256gcm",
            budget,
            &stringzilla_gcm,
            &plaintext_lengths,
            |index, ciphertext, tag| {
                let nonce = stringzilla_nonce_for_index(index);
                let verdict = key_stringzilla_gcm.decrypt_into(
                    &nonce,
                    &[],
                    ciphertext,
                    &mut opened[..ciphertext.len()],
                    tag,
                );
                let _ = black_box(verdict);
            },
        );
    }

    // Benchmark: StringZilla AES-256-CTR decryption, the same call as encryption because counter
    // mode is its own inverse. It carries no tag, so nothing is authenticated here.
    {
        let key = Aes256CtrKey::new(&key_bytes);
        let mut opened = vec![0u8; stringzilla_gcm.longest()];
        measure_per_ciphertext(
            "decryption/stringzilla::aes256ctr",
            budget,
            &stringzilla_gcm,
            &plaintext_lengths,
            |index, ciphertext, _| {
                let nonce = stringzilla_nonce_for_index(index);
                key.xor_into(&nonce, 0, ciphertext, &mut opened[..ciphertext.len()]);
                let _ = black_box(&opened[..ciphertext.len()]);
            },
        );
    }

    let key_sodium_chacha = SodiumChaCha20Key([0u8; chacha20poly1305_ietf::KEYBYTES]);
    let sodium_chacha = SealedCorpus::new(tokens, |index, token, ciphertext| {
        ciphertext.extend_from_slice(token);
        chacha20poly1305_ietf::seal_detached(
            ciphertext,
            None,
            &sodium_chacha20_nonce_for_index(index),
            &key_sodium_chacha,
        )
    });

    // Benchmark: libsodium ChaCha20-Poly1305 IETF decryption
    {
        let mut opened = vec![0u8; sodium_chacha.longest()];
        measure_per_ciphertext(
            "decryption/libsodium::chacha20poly1305_ietf",
            budget,
            &sodium_chacha,
            &plaintext_lengths,
            |index, ciphertext, tag| {
                let message = &mut opened[..ciphertext.len()];
                message.copy_from_slice(ciphertext);
                let nonce = sodium_chacha20_nonce_for_index(index);
                let verdict = chacha20poly1305_ietf::open_detached(
                    message,
                    None,
                    tag,
                    &nonce,
                    &key_sodium_chacha,
                );
                let _ = black_box(verdict);
            },
        );
    }

    let key_sodium_xchacha = SodiumXChaCha20Key([0u8; xchacha20poly1305_ietf::KEYBYTES]);
    let sodium_xchacha = SealedCorpus::new(tokens, |index, token, ciphertext| {
        ciphertext.extend_from_slice(token);
        xchacha20poly1305_ietf::seal_detached(
            ciphertext,
            None,
            &sodium_xchacha20_nonce_for_index(index),
            &key_sodium_xchacha,
        )
    });

    // Benchmark: libsodium XChaCha20-Poly1305 IETF decryption
    {
        let mut opened = vec![0u8; sodium_xchacha.longest()];
        measure_per_ciphertext(
            "decryption/libsodium::xchacha20poly1305_ietf",
            budget,
            &sodium_xchacha,
            &plaintext_lengths,
            |index, ciphertext, tag| {
                let message = &mut opened[..ciphertext.len()];
                message.copy_from_slice(ciphertext);
                let nonce = sodium_xchacha20_nonce_for_index(index);
                let verdict = xchacha20poly1305_ietf::open_detached(
                    message,
                    None,
                    tag,
                    &nonce,
                    &key_sodium_xchacha,
                );
                let _ = black_box(verdict);
            },
        );
    }
}

fn main() {
    install_panic_hook();
    log_stringzilla_metadata();

    // Initialize libsodium
    sodiumoxide::init().expect("Failed to initialize libsodium");

    // Load the dataset defined by the environment variables
    let tape = load_dataset().unwrap_nice();

    let budget = BenchBudget::from_env(5.0, 10.0);

    // Profile key generation and cipher initialization overhead
    println!("# keygen");
    bench_key_generation(&budget);

    // Profile encryption operations
    println!("# encryption");
    bench_encryption(&budget, &tape);

    // Profile decryption operations
    println!("# decryption");
    bench_decryption(&budget, &tape);
}
