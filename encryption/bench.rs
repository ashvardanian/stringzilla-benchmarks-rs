#![doc = r#"# StringWars: Encryption

AEAD benchmarks: key generation, encryption and decryption across ring, OpenSSL and libsodium.

## System Dependencies

```sh
sudo apt install -y libssl-dev libsodium-dev   # Ubuntu/Debian
sudo dnf install -y openssl-devel libsodium-devel  # RHEL/Fedora
brew install openssl libsodium                 # macOS
```

```sh
STRINGWARS_DATASET=README.md cargo bench --features bench_encryption --bench bench_encryption
```
"#]
use std::hint::black_box;

use openssl::symm::{Cipher, Crypter, Mode};
use stringtape::BytesCowsAuto;

use stringwars::{
    expect_ok, finish, install_panic_hook, log_stringzilla_metadata, log_timing_overhead, measure,
    resolve_dataset, MeasureSpec, ResultExt, Unit, WorkUnits,
};

/// The nonce bytes a given token is sealed under: the index little-endian in the
/// leading 8, zero elsewhere.
///
/// Indexed, never counted. A counter living outside the pass closure keeps advancing
/// across passes, so pass 2 opens ciphertext 0 with a nonce it was never sealed
/// under and every decrypt row fails - invisibly, while the `Result` was discarded.
fn nonce_bytes<const N: usize>(index: u64) -> [u8; N] {
    let mut bytes = [0u8; N];
    bytes[..8].copy_from_slice(&index.to_le_bytes());
    bytes
}

fn ring_nonce_at(index: u64) -> ring::aead::Nonce {
    ring::aead::Nonce::assume_unique_for_key(nonce_bytes(index))
}

fn next_ring_nonce(counter: &mut u64) -> ring::aead::Nonce {
    *counter = counter.wrapping_add(1);
    ring_nonce_at(*counter - 1)
}

fn openssl_iv_at(index: u64) -> [u8; 12] {
    nonce_bytes(index)
}

fn next_openssl_iv(counter: &mut u64) -> [u8; 12] {
    *counter = counter.wrapping_add(1);
    openssl_iv_at(*counter - 1)
}

fn sodium_chacha20_nonce_at(index: u64) -> sodiumoxide::crypto::aead::chacha20poly1305_ietf::Nonce {
    sodiumoxide::crypto::aead::chacha20poly1305_ietf::Nonce(nonce_bytes(index))
}

fn next_sodium_chacha20_nonce(
    counter: &mut u64,
) -> sodiumoxide::crypto::aead::chacha20poly1305_ietf::Nonce {
    *counter = counter.wrapping_add(1);
    sodium_chacha20_nonce_at(*counter - 1)
}

fn sodium_xchacha20_nonce_at(
    index: u64,
) -> sodiumoxide::crypto::aead::xchacha20poly1305_ietf::Nonce {
    sodiumoxide::crypto::aead::xchacha20poly1305_ietf::Nonce(nonce_bytes(index))
}

fn next_sodium_xchacha20_nonce(
    counter: &mut u64,
) -> sodiumoxide::crypto::aead::xchacha20poly1305_ietf::Nonce {
    *counter = counter.wrapping_add(1);
    sodium_xchacha20_nonce_at(*counter - 1)
}

/// Seals `plaintext` into `out`, writing the detached tag, and returns the ciphertext length.
///
/// OpenSSL's `encrypt_aead`/`decrypt_aead` and libsodium's `seal`/`open` return an owned `Vec`
/// per message; the `Crypter` and detached forms used here write into a reused buffer, so all
/// twelve cipher rows are about the cipher rather than the allocator.
fn openssl_seal(
    cipher: Cipher,
    key: &[u8],
    iv: &[u8],
    plaintext: &[u8],
    out: &mut [u8],
    tag: &mut [u8],
) -> usize {
    let mut crypter = expect_ok(Crypter::new(cipher, Mode::Encrypt, key, Some(iv)));
    let count = expect_ok(crypter.update(plaintext, out));
    let rest = expect_ok(crypter.finalize(&mut out[count..]));
    expect_ok(crypter.get_tag(tag));
    count + rest
}

/// Verifies `tag` and opens `ciphertext` into `out`, returning the plaintext length.
fn openssl_open(
    cipher: Cipher,
    key: &[u8],
    iv: &[u8],
    ciphertext: &[u8],
    out: &mut [u8],
    tag: &[u8],
) -> usize {
    let mut crypter = expect_ok(Crypter::new(cipher, Mode::Decrypt, key, Some(iv)));
    let count = expect_ok(crypter.update(ciphertext, out));
    expect_ok(crypter.set_tag(tag));
    let rest = expect_ok(crypter.finalize(&mut out[count..]));
    count + rest
}

/// Benchmarks key generation and cipher setup overhead. Each variant builds one key/cipher per
/// call and cycles for the budget; throughput is reported as bytes/s over the 32-byte key.
fn bench_key_generation() {
    use ring::aead;

    measure(
        "keygen/ring::chacha20poly1305",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(32)),
        || {
            let key_bytes = [0u8; 32]; // 256-bit key
            let key = aead::UnboundKey::new(&aead::CHACHA20_POLY1305, &key_bytes);
            let _ = black_box(key);
        },
    );

    measure(
        "keygen/ring::aes256gcm",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(32)),
        || {
            let key_bytes = [0u8; 32]; // 256-bit key
            let key = aead::UnboundKey::new(&aead::AES_256_GCM, &key_bytes);
            let _ = black_box(key);
        },
    );

    {
        measure(
            "keygen/openssl::chacha20poly1305",
            MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(32)),
            || {
                let key = [0u8; 32];
                // ChaCha20-Poly1305 is an AEAD, so OpenSSL requires the 12-byte IV up front; passing `None`
                // fails with "an IV is required for this cipher" on current OpenSSL builds.
                let iv = [0u8; 12];
                let cipher = Cipher::chacha20_poly1305();
                let crypter = Crypter::new(cipher, Mode::Encrypt, &key, Some(&iv));
                let _ = black_box(crypter);
            },
        );
    }

    {
        measure(
            "keygen/openssl::aes256gcm",
            MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(32)),
            || {
                let key = [0u8; 32];
                // AES-256-GCM likewise needs its 12-byte IV at construction time (see the ChaCha20 note above).
                let iv = [0u8; 12];
                let cipher = Cipher::aes_256_gcm();
                let crypter = Crypter::new(cipher, Mode::Encrypt, &key, Some(&iv));
                let _ = black_box(crypter);
            },
        );
    }

    {
        use sodiumoxide::crypto::aead::chacha20poly1305_ietf::{self, Key};
        measure(
            "keygen/libsodium::chacha20poly1305_ietf",
            MeasureSpec::new(
                Unit::Bytes,
                WorkUnits::bytes(chacha20poly1305_ietf::KEYBYTES as u64),
            ),
            || {
                let key = Key([0u8; chacha20poly1305_ietf::KEYBYTES]);
                let _ = black_box(key);
            },
        );
    }

    {
        use sodiumoxide::crypto::aead::xchacha20poly1305_ietf::{self, Key};
        measure(
            "keygen/libsodium::xchacha20poly1305_ietf",
            MeasureSpec::new(
                Unit::Bytes,
                WorkUnits::bytes(xchacha20poly1305_ietf::KEYBYTES as u64),
            ),
            || {
                let key = Key([0u8; xchacha20poly1305_ietf::KEYBYTES]);
                let _ = black_box(key);
            },
        );
    }
}

/// Benchmarks AEAD encryption (encrypt + authenticate). Each variant encrypts one token per call
/// and cycles the dataset for the budget; throughput is reported as bytes/s over the plaintext.
fn bench_encryption(tokens: &BytesCowsAuto) {
    use ring::aead::{self, Aad, LessSafeKey, UnboundKey};

    // Collect token slices once so each cyclic call indexes a single token.
    let slices: Vec<&[u8]> = tokens.iter().collect();
    let longest_token = slices.iter().map(|token| token.len()).max().unwrap_or(0);
    let pass_work = WorkUnits::new(
        slices.len() as u64,
        slices.iter().map(|token| token.len() as u64).sum(),
    );

    {
        let key_bytes = [0u8; 32];
        let unbound_key = UnboundKey::new(&aead::CHACHA20_POLY1305, &key_bytes).unwrap();
        let key = LessSafeKey::new(unbound_key);

        // One scratch buffer, sized for the longest token plus its tag: `to_vec()`
        // allocates capacity == len, so a later `reserve` always reallocates.
        let mut in_out = Vec::with_capacity(longest_token + aead::CHACHA20_POLY1305.tag_len());
        let mut nonce_counter: u64 = 0;
        measure(
            "encryption/ring::chacha20poly1305",
            MeasureSpec::new(Unit::Bytes, pass_work),
            || {
                for token in slices.iter() {
                    in_out.clear();
                    in_out.extend_from_slice(token);
                    let nonce = next_ring_nonce(&mut nonce_counter);
                    let _ =
                        black_box(key.seal_in_place_append_tag(nonce, Aad::empty(), &mut in_out));
                }
            },
        );
    }

    {
        let key_bytes = [0u8; 32];
        let unbound_key = UnboundKey::new(&aead::AES_256_GCM, &key_bytes).unwrap();
        let key = LessSafeKey::new(unbound_key);

        // One scratch buffer, sized for the longest token plus its tag: `to_vec()`
        // allocates capacity == len, so a later `reserve` always reallocates.
        let mut in_out = Vec::with_capacity(longest_token + aead::AES_256_GCM.tag_len());
        let mut nonce_counter: u64 = 0;
        measure(
            "encryption/ring::aes256gcm",
            MeasureSpec::new(Unit::Bytes, pass_work),
            || {
                for token in slices.iter() {
                    in_out.clear();
                    in_out.extend_from_slice(token);
                    let nonce = next_ring_nonce(&mut nonce_counter);
                    let _ =
                        black_box(key.seal_in_place_append_tag(nonce, Aad::empty(), &mut in_out));
                }
            },
        );
    }

    {
        let key = [0u8; 32];
        let cipher = Cipher::chacha20_poly1305();
        let mut ciphertext = vec![0u8; longest_token + cipher.block_size()];
        let mut nonce_counter: u64 = 0;
        measure(
            "encryption/openssl::chacha20poly1305",
            MeasureSpec::new(Unit::Bytes, pass_work),
            || {
                for token in slices.iter() {
                    let iv = next_openssl_iv(&mut nonce_counter);
                    let mut tag = [0u8; 16];
                    black_box(openssl_seal(
                        cipher,
                        &key,
                        &iv,
                        token,
                        &mut ciphertext,
                        &mut tag,
                    ));
                }
            },
        );
    }

    {
        let key = [0u8; 32];
        let cipher = Cipher::aes_256_gcm();
        let mut ciphertext = vec![0u8; longest_token + cipher.block_size()];
        let mut nonce_counter: u64 = 0;
        measure(
            "encryption/openssl::aes256gcm",
            MeasureSpec::new(Unit::Bytes, pass_work),
            || {
                for token in slices.iter() {
                    let iv = next_openssl_iv(&mut nonce_counter);
                    let mut tag = [0u8; 16];
                    black_box(openssl_seal(
                        cipher,
                        &key,
                        &iv,
                        token,
                        &mut ciphertext,
                        &mut tag,
                    ));
                }
            },
        );
    }

    {
        use sodiumoxide::crypto::aead::chacha20poly1305_ietf::{self, Key};

        let key = Key([0u8; chacha20poly1305_ietf::KEYBYTES]);
        let mut in_out = Vec::with_capacity(longest_token);
        let mut nonce_counter: u64 = 0;
        measure(
            "encryption/libsodium::chacha20poly1305_ietf",
            MeasureSpec::new(Unit::Bytes, pass_work),
            || {
                for token in slices.iter() {
                    in_out.clear();
                    in_out.extend_from_slice(token);
                    let nonce = next_sodium_chacha20_nonce(&mut nonce_counter);
                    let _ = black_box(chacha20poly1305_ietf::seal_detached(
                        &mut in_out,
                        None,
                        &nonce,
                        &key,
                    ));
                }
            },
        );
    }

    {
        use sodiumoxide::crypto::aead::xchacha20poly1305_ietf::{self, Key};

        let key = Key([0u8; xchacha20poly1305_ietf::KEYBYTES]);
        let mut in_out = Vec::with_capacity(longest_token);
        let mut nonce_counter: u64 = 0;
        measure(
            "encryption/libsodium::xchacha20poly1305_ietf",
            MeasureSpec::new(Unit::Bytes, pass_work),
            || {
                for token in slices.iter() {
                    in_out.clear();
                    in_out.extend_from_slice(token);
                    let nonce = next_sodium_xchacha20_nonce(&mut nonce_counter);
                    let _ = black_box(xchacha20poly1305_ietf::seal_detached(
                        &mut in_out,
                        None,
                        &nonce,
                        &key,
                    ));
                }
            },
        );
    }
}

/// Benchmarks AEAD decryption (verify + decrypt). Pre-encrypts every token once before the timed
/// loop; each measured call then decrypts one ciphertext and cycles the dataset for the budget.
/// Throughput is reported over the original plaintext lengths to match the encryption accounting.
fn bench_decryption(tokens: &BytesCowsAuto) {
    use ring::aead::{self, Aad, LessSafeKey, UnboundKey};

    // Byte work is the original plaintext: the sealed blobs carry tag bytes the encryption
    // accounting excluded too.
    let decrypt_pass_work = WorkUnits::new(
        tokens.len() as u64,
        tokens.iter().map(|token| token.len() as u64).sum(),
    );
    let longest_token = tokens.iter().map(|token| token.len()).max().unwrap_or(0);

    let key_bytes = [0u8; 32];
    let unbound_key_chacha = UnboundKey::new(&aead::CHACHA20_POLY1305, &key_bytes).unwrap();
    let key_chacha = LessSafeKey::new(unbound_key_chacha);
    let mut encrypted_tokens_chacha: Vec<Vec<u8>> = Vec::new();
    {
        let mut nonce_counter: u64 = 0;
        for token in tokens.iter() {
            let mut in_out = token.to_vec();
            in_out.reserve(aead::CHACHA20_POLY1305.tag_len());
            let nonce = next_ring_nonce(&mut nonce_counter);
            key_chacha
                .seal_in_place_append_tag(nonce, Aad::empty(), &mut in_out)
                .unwrap();
            encrypted_tokens_chacha.push(in_out);
        }
    }

    let unbound_key_aes = UnboundKey::new(&aead::AES_256_GCM, &key_bytes).unwrap();
    let key_aes = LessSafeKey::new(unbound_key_aes);
    let mut encrypted_tokens_aes: Vec<Vec<u8>> = Vec::new();
    {
        let mut nonce_counter: u64 = 0;
        for token in tokens.iter() {
            let mut in_out = token.to_vec();
            in_out.reserve(aead::AES_256_GCM.tag_len());
            let nonce = next_ring_nonce(&mut nonce_counter);
            key_aes
                .seal_in_place_append_tag(nonce, Aad::empty(), &mut in_out)
                .unwrap();
            encrypted_tokens_aes.push(in_out);
        }
    }

    {
        let mut in_out = Vec::with_capacity(
            encrypted_tokens_chacha
                .iter()
                .map(Vec::len)
                .max()
                .unwrap_or(0),
        );
        measure(
            "decryption/ring::chacha20poly1305",
            MeasureSpec::new(Unit::Bytes, decrypt_pass_work),
            || {
                for (index, blob) in encrypted_tokens_chacha.iter().enumerate() {
                    in_out.clear();
                    in_out.extend_from_slice(blob);
                    let nonce = ring_nonce_at(index as u64);
                    black_box(expect_ok(key_chacha.open_in_place(
                        nonce,
                        Aad::empty(),
                        &mut in_out,
                    )));
                }
            },
        );
    }

    {
        let mut in_out =
            Vec::with_capacity(encrypted_tokens_aes.iter().map(Vec::len).max().unwrap_or(0));
        measure(
            "decryption/ring::aes256gcm",
            MeasureSpec::new(Unit::Bytes, decrypt_pass_work),
            || {
                for (index, blob) in encrypted_tokens_aes.iter().enumerate() {
                    in_out.clear();
                    in_out.extend_from_slice(blob);
                    let nonce = ring_nonce_at(index as u64);
                    black_box(expect_ok(key_aes.open_in_place(
                        nonce,
                        Aad::empty(),
                        &mut in_out,
                    )));
                }
            },
        );
    }

    use openssl::symm::encrypt_aead;
    let cipher_chacha = Cipher::chacha20_poly1305();
    let mut encrypted_tokens_openssl_chacha: Vec<(Vec<u8>, [u8; 16])> = Vec::new();
    {
        let mut nonce_counter: u64 = 0;
        for token in tokens.iter() {
            let iv = next_openssl_iv(&mut nonce_counter);
            let mut tag = [0u8; 16];
            let ciphertext =
                encrypt_aead(cipher_chacha, &key_bytes, Some(&iv), &[], token, &mut tag).unwrap();
            encrypted_tokens_openssl_chacha.push((ciphertext, tag));
        }
    }

    let cipher_aes = Cipher::aes_256_gcm();
    let mut encrypted_tokens_openssl_aes: Vec<(Vec<u8>, [u8; 16])> = Vec::new();
    {
        let mut nonce_counter: u64 = 0;
        for token in tokens.iter() {
            let iv = next_openssl_iv(&mut nonce_counter);
            let mut tag = [0u8; 16];
            let ciphertext =
                encrypt_aead(cipher_aes, &key_bytes, Some(&iv), &[], token, &mut tag).unwrap();
            encrypted_tokens_openssl_aes.push((ciphertext, tag));
        }
    }

    {
        let mut plaintext = vec![0u8; longest_token + cipher_chacha.block_size()];
        measure(
            "decryption/openssl::chacha20poly1305",
            MeasureSpec::new(Unit::Bytes, decrypt_pass_work),
            || {
                for (index, (ciphertext, tag)) in encrypted_tokens_openssl_chacha.iter().enumerate()
                {
                    let iv = openssl_iv_at(index as u64);
                    black_box(openssl_open(
                        cipher_chacha,
                        &key_bytes,
                        &iv,
                        ciphertext,
                        &mut plaintext,
                        tag,
                    ));
                }
            },
        );
    }

    {
        let mut plaintext = vec![0u8; longest_token + cipher_aes.block_size()];
        measure(
            "decryption/openssl::aes256gcm",
            MeasureSpec::new(Unit::Bytes, decrypt_pass_work),
            || {
                for (index, (ciphertext, tag)) in encrypted_tokens_openssl_aes.iter().enumerate() {
                    let iv = openssl_iv_at(index as u64);
                    black_box(openssl_open(
                        cipher_aes,
                        &key_bytes,
                        &iv,
                        ciphertext,
                        &mut plaintext,
                        tag,
                    ));
                }
            },
        );
    }

    use sodiumoxide::crypto::aead::chacha20poly1305_ietf::{self, Key as SodiumChaCha20Key};
    let key_sodium_chacha = SodiumChaCha20Key([0u8; chacha20poly1305_ietf::KEYBYTES]);
    let mut encrypted_tokens_sodium_chacha: Vec<(Vec<u8>, chacha20poly1305_ietf::Tag)> = Vec::new();
    {
        let mut nonce_counter: u64 = 0;
        for token in tokens.iter() {
            let mut ciphertext = token.to_vec();
            let nonce = next_sodium_chacha20_nonce(&mut nonce_counter);
            let tag = chacha20poly1305_ietf::seal_detached(
                &mut ciphertext,
                None,
                &nonce,
                &key_sodium_chacha,
            );
            encrypted_tokens_sodium_chacha.push((ciphertext, tag));
        }
    }

    {
        let mut in_out = Vec::with_capacity(longest_token);
        measure(
            "decryption/libsodium::chacha20poly1305_ietf",
            MeasureSpec::new(Unit::Bytes, decrypt_pass_work),
            || {
                for (index, (ciphertext, tag)) in encrypted_tokens_sodium_chacha.iter().enumerate()
                {
                    in_out.clear();
                    in_out.extend_from_slice(ciphertext);
                    let nonce = sodium_chacha20_nonce_at(index as u64);
                    expect_ok(chacha20poly1305_ietf::open_detached(
                        &mut in_out,
                        None,
                        tag,
                        &nonce,
                        &key_sodium_chacha,
                    ));
                    black_box(&in_out);
                }
            },
        );
    }

    use sodiumoxide::crypto::aead::xchacha20poly1305_ietf::{self, Key as SodiumXChaCha20Key};
    let key_sodium_xchacha = SodiumXChaCha20Key([0u8; xchacha20poly1305_ietf::KEYBYTES]);
    let mut encrypted_tokens_sodium_xchacha: Vec<(Vec<u8>, xchacha20poly1305_ietf::Tag)> =
        Vec::new();
    {
        let mut nonce_counter: u64 = 0;
        for token in tokens.iter() {
            let mut ciphertext = token.to_vec();
            let nonce = next_sodium_xchacha20_nonce(&mut nonce_counter);
            let tag = xchacha20poly1305_ietf::seal_detached(
                &mut ciphertext,
                None,
                &nonce,
                &key_sodium_xchacha,
            );
            encrypted_tokens_sodium_xchacha.push((ciphertext, tag));
        }
    }

    {
        let mut in_out = Vec::with_capacity(longest_token);
        measure(
            "decryption/libsodium::xchacha20poly1305_ietf",
            MeasureSpec::new(Unit::Bytes, decrypt_pass_work),
            || {
                for (index, (ciphertext, tag)) in encrypted_tokens_sodium_xchacha.iter().enumerate()
                {
                    in_out.clear();
                    in_out.extend_from_slice(ciphertext);
                    let nonce = sodium_xchacha20_nonce_at(index as u64);
                    expect_ok(xchacha20poly1305_ietf::open_detached(
                        &mut in_out,
                        None,
                        tag,
                        &nonce,
                        &key_sodium_xchacha,
                    ));
                    black_box(&in_out);
                }
            },
        );
    }
}

fn main() {
    install_panic_hook();
    log_stringzilla_metadata();

    sodiumoxide::init().expect("Failed to initialize libsodium");

    let tape = resolve_dataset("encryption").unwrap_nice();
    log_timing_overhead();

    // Profile key generation and cipher initialization overhead
    println!("# keygen");
    bench_key_generation();

    // Profile encryption operations
    println!("# encryption");
    bench_encryption(&tape);

    // Profile decryption operations
    println!("# decryption");
    bench_decryption(&tape);

    finish();
}
