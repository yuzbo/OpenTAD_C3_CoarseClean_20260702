from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable


HASH_SUITE_ID = "DUCA-ADMISSION-V2.1-HASH-SUITE-V1"
PROTOCOL_ID = "DUCA-ADMISSION-V2.1-REALVIDEO-CROSSED-NULL"


def canonical_text(value: str, *, field_name: str = "text") -> bytes:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if "\x00" in value:
        raise ValueError(f"{field_name} must not contain NUL")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{field_name} must already be Unicode NFC")
    return value.encode("utf-8", errors="strict")


def u8(value: int) -> bytes:
    if type(value) is not int:
        raise TypeError("u8 value must be an integer")
    if not 0 <= value < 1 << 8:
        raise ValueError("u8 value is out of range")
    return value.to_bytes(1, "big")


def u32be(value: int) -> bytes:
    if type(value) is not int:
        raise TypeError("u32 value must be an integer")
    if not 0 <= value < 1 << 32:
        raise ValueError("u32 value is out of range")
    return value.to_bytes(4, "big")


def u64be(value: int) -> bytes:
    if type(value) is not int:
        raise TypeError("u64 value must be an integer")
    if not 0 <= value < 1 << 64:
        raise ValueError("u64 value is out of range")
    return value.to_bytes(8, "big")


def lp(raw: bytes) -> bytes:
    if not isinstance(raw, bytes):
        raise TypeError("length-prefixed values must be bytes")
    return u64be(len(raw)) + raw


def _domain_bytes(domain: str) -> bytes:
    if not isinstance(domain, str) or not domain:
        raise ValueError("hash domain must be a non-empty ASCII string")
    try:
        return domain.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("hash domain must be ASCII") from exc


def domain_hash(domain: str, *fields: bytes) -> bytes:
    transcript = b"".join(
        (
            lp(HASH_SUITE_ID.encode("ascii")),
            lp(PROTOCOL_ID.encode("ascii")),
            lp(_domain_bytes(domain)),
            *(lp(field) for field in fields),
        )
    )
    return hashlib.sha256(transcript).digest()


def domain_hash_hex(domain: str, *fields: bytes) -> str:
    return domain_hash(domain, *fields).hex()


def raw_sha256(value: bytes) -> str:
    if not isinstance(value, bytes):
        raise TypeError("raw SHA-256 input must be bytes")
    return hashlib.sha256(value).hexdigest()


def sha256_bytes(value: str, *, field_name: str = "sha256") -> bytes:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field_name} must contain 64 lowercase hex characters")
    if value.lower() != value:
        raise ValueError(f"{field_name} must be lowercase")
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} is not hexadecimal") from exc


def draw_mod(m: int, domain: str, *fields: bytes) -> int:
    if type(m) is not int:
        raise TypeError("draw modulus must be an integer")
    if m < 2 or m >= 1 << 64:
        raise ValueError("draw modulus must be in [2, 2^64)")
    limit = (1 << 64) - ((1 << 64) % m)
    counter = 0
    while True:
        digest = domain_hash(domain, *fields, u64be(counter))
        value = int.from_bytes(digest[:8], "big")
        if value < limit:
            return value % m
        counter += 1


def join_hash_fields(values: Iterable[str], *, field_name: str) -> tuple[bytes, ...]:
    return tuple(canonical_text(value, field_name=field_name) for value in values)
