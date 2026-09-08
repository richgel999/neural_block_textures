# NTCB plan: a bit-packed container for the codec, round-tripped through both decoders

Plan written September 7, 2026 by a planning agent; line numbers refer to the tree at commit aa52197 (`main.cpp` 3231 lines, `ntc_decode.cpp` 1217 lines). Purpose: a real file, no entropy coding, whose byte count is the codec's bitrate; the trainer writes it, `ntc_decode` and the trainer's loader read it, and the reconstruction is byte-identical to the `model.bin` path. It is the system sanity check for every bitrate figure quoted so far, and the container that a real entropy coder drops into later (section 5).

Decisions taken for the open questions (September 7, 2026): 1 = round the final weights to fp16 under the flag; 2 = install the file's grid as a frozen `--qes B` level on reload; 3 = stay explicit (`--write-ntcb`); 4 = keep the simulator's unsigned DC width; 5 = no per-channel alignment inside a level body.

## 0. Decisions taken in this plan (each reversible; flagged where Richard may want otherwise)

1. **Explicit flag, not automatic.** `--write-ntcb` (off by default) writes `<out>/model.ntcb` once, at the end of the run (or of a `--load ... --iters 0` evaluation). Reason: writing the file has one side effect on the run's numbers (item 2), so it must be opt-in; also it keeps every existing `--dct-q` / `--qat` log byte-identical.
2. **The MLP is really fp16 when the file is written.** `bitrate_stats` already charges 16 bits per weight (`s.bits_mlp = m.size() * 16.0`, main.cpp L1925) but every decode, `recon_q_final.png` and the `done:` PSNR use the fp32 weights (L3106, L3169). A file that stores fp16 cannot round-trip to that output. So with `--write-ntcb` the weights are rounded through fp16 (round-to-nearest-even) once, at the last iteration right after the final device download (L3094) and before the last decode (L3096); for `--iters 0` right after the loop, before `--resave` (L3163). Then the last progress line, `model.bin` (L3159 / L3163), `recon_q_final.png`, the `done:` PSNR and the `.ntcb` all describe the same weights, and (b), (c) below become exact equalities. The rounding is printed (`ntcb     : mlp weights rounded to fp16 (N of M changed, max |dw| ...)`). Alternative if this is unwanted: store fp16, and verify (b) against `ntc_decode model.bin --mlp-fp16` instead; then (c) can only hold to display precision. The first is adopted.
3. **Per-level coding modes** in the file, not only a level-0 mode: `0 = q8` (post-hoc `--qbits` grid: per-channel fp32 lo/hi + B-bit indices), `1 = qes` (same wire layout, grid `lo + k/(float)levels*range`), `2 = qat` (per-channel palette of 2^B fp32 values + B-bit indices), `3 = dct` (symbols). Level 0 may be any of the four (the `c1float`, `c1qes8`, `c1q4`, `dct50` runs all exist); levels >= 1 are 0 or 1. The qes variant of "continuous-8bit" needs its own tag only so the reader knows which rounding produced the index (both dequantize with the same expression, section 1.3).
4. **qat palette is stored, not recomputed.** CPU_DECODER_NOTES.md deviation 1: the trainer's grid values (`QAT_GRID`, L1705-1707, `/fp:fast`) are not the strict `-1 + 2k/levels`; ntc_decode reproduces them only through a LUT built from the data. The file carries 2^B fp32 per channel (64 bytes at B = 4). The simulator does not charge it (a `fixed` level pays no header, L1978), so it is counted as container side-info in the reconciliation.
5. **Bit order** LSB-first within a little-endian byte stream (a 64-bit accumulator, `put(v, n)` shifts in at the top, bytes emitted from the bottom); all multi-byte header fields little-endian written byte by byte, so nothing depends on host endianness. Every section byte-aligned with zero padding; no alignment inside a section.
6. **Both readers construct the existing in-memory state** (`Decoder` in main.cpp, `Model` in ntc_decode.cpp) and then use the untouched decode paths: `dct_recon_level0` (main.cpp L796-803, ntc_decode.cpp L425-459) for the IDCT, `qes_rebuild` (L410-418) for grids, the SIMD decoder as is.

## 1. Byte-exact layout, `ntcb` container version 1

### 1.1 Header (all integers little-endian; offsets in bytes)

**Amendment, September 7, 2026 (review fixes; the container is uncommitted, so the layout changed in place and the version stays 1).** Every layout change relative to the table as first written: (1) `channels_per_texture` is now T bytes, one per texture (each 3 today, `nout` = their sum; a per-texture channel count later needs no version bump), placed after `nout` at offset 52; the fixed fields 32..39 are now `latent_cell, dct_block, T, nlevels, activation, clamp_out, nhidden, reserved (0)`. (2) The hash at offset 28 covers the **whole file** with the hash field itself read as zero (`ntcb_file_hash`), so the header is covered too; it is checked right after `file_bytes`. (3) **Sync markers**: 6-bit constants inside the section bodies (section 1.2): `NTCB_SYNC_LEVEL 0x2B` at the start of every level body, `NTCB_SYNC_CHANNEL 0x16` at the start of every channel of a DCT body (before its scale codes), `NTCB_SYNC_MLP 0x31` at the start of the MLP body, `NTCB_SYNC_END 0x0C` at the end of every level body and of the MLP body (pairwise Hamming distance >= 3, none all-zero or all-one). Each written value is `(base + 13 * level + 7 * channel) & 63` (`ntcb_sync_value`, mirrored in ntc_decode.cpp; the MLP markers use level = nlevels), so the same kind of marker carries a different value in every level and channel (13l + 7c is distinct for every level 0..2, channel 0..3 pair) and a reader that reaches the right kind of marker in the wrong section still refuses (his request, September 7, 2026). They are container overhead, not content bits (the `sync N bits` term of both file lines), so the reconciliation `content_bits == sim_total - side` is unchanged; a two-level DCT file carries 7 markers (42 bits). Both readers check every marker and refuse with `sync marker mismatch before level l body` / `... before level l channel c` / `... after level l body` / `... before the mlp body` / `... after the mlp body`. (4) Both readers compute the fewest bytes the header's sections can occupy (64-bit: every grid level's `W*H*sum(bits)`, a DCT level's scale codes + DC + EOB per block, the MLP's 16 per weight, the markers, 8 per section header) and refuse a header that promises more than `file_bytes - H` before anything is allocated from its fields; level dims are capped at 2^20 and the decode size at 16384 x 16384. (5) The MLP section's `coding` byte 0 means raw IEEE binary16, little-endian, one per weight, in `D.mlp.p` order (defined explicitly; 1 and 2 are reserved as in section 5). (6) `latent_cell` and `dct_block` are informational: neither reader derives anything from them beyond the consistency check `dct_block == 8` on a DCT level and `0` otherwise (the trainer decodes at the run's own size and cell; ntc_decode reads the level records). The mode-0 index and value are now computed by the strict-FP helpers `q8_index` / `q8_value` in both `bitrate_stats` and the writer (section 1.3).

**Amendment, September 7, 2026 (evening; `--dct-deadzone`, container version 2).** A mode-3 level record gains one byte `dct_flags` right after `dc_step` and before the C quality bytes: bit 0 = 1 means the symbols were produced by the dead-zone AC quantizer (alpha 0.5, first-order exemption: the only quantizer before this amendment), 0 means plain rounding on every AC (`--dct-deadzone 0`); bits 1..7 are reserved and both readers refuse a record with any of them set. The header `version` field becomes 2; the writer always writes 2; both readers accept 1 (no `dct_flags` byte, dead zone implied) and 2, and refuse anything else (`ntcb version N (this build reads versions 1 and 2)`). Nothing else in the layout changes: the payload, the sync markers, the reconciliation and the hash rule are as before, so a version-1 file written before the amendment still loads and decodes byte for byte (`out_model16_b8_dct80_c4_mlp27_lam2_8k/model.ntcb`, DCT_NOTES.md section 9). The trainer's loader checks the flag against `--dct-deadzone` like `dc_step` against `--dct-dc-step`; `ntc_decode` dequantizes with the file's flag. The model file has the same addition as v14 (one int after the three DCT header ints).

```
off  size            field
0    4               magic: bytes 6E 74 63 62 ("ntcb"); as a LE uint32 0x6263746E (distinct from 0x4E5443xx)
4    2               version = 2 (1 before the --dct-deadzone amendment; both readers accept 1 and 2)
6    2               header_bytes H (the sections begin at H; the reader refuses H > file size)
8    4               img_w   (decoded / padded width  = D.W;  ntc_decode m.imgW; at most 16384)
12   4               img_h
16   4               src_w   (source before --block padding = o.orig_w; ntc_decode m.srcW)
20   4               src_h
24   4               file_bytes (total; refused if != actual size: truncation check)
28   4               file_hash: FNV-1a 32 over bytes [0, file_bytes) with these four bytes read as zero (corruption check of the whole file, header included)
32   1               latent_cell (o.block; 0 = no --block; informational; the writer refuses --block > 255)
33   1               dct_block (8 when level 0 is dct, else 0; informational beyond that consistency check)
34   1               T textures (1..4)
35   1               nlevels (1..3)
36   1               activation (0 = leaky; anything else refused)
37   1               clamp_out (0 sigmoid, 1 clamp)
38   1               nhidden (1..8)
39   1               reserved (0; refused otherwise)
40   4               leak (fp32 bits)
44   4               nin
48   4               nout (= the sum of the channel counts below; checked)
52   T               channels_per_texture[t] (3 each today; a reader refuses anything else)
..   2*nhidden       hidden widths, uint16 each
..   2               poslen, then poslen bytes of the positional spec text (exactly D.pos.spec, as save_model writes it, L1849)
per level l = 0..nlevels-1:
     4 W, 4 H, 2 C, 1 filter (0 bilinear / 1 nearest), 1 mode (0 q8 / 1 qes / 2 qat / 3 dct)
     C bytes  bits[c]     (mode 0/1: B in 2..12 (8 for q8); mode 2: B in 1..8; mode 3: 0)
     mode 3 only: 1 byte dc_step (1..64), [version 2] 1 byte dct_flags (bit 0 = dead-zone quantizer; bits 1..7 must be 0), C bytes q[c] (1..100)
     side info:
        mode 0/1: C x (fp32 lo, fp32 hi)                 = 64 bits per channel, the simulator's header charge (L1978)
        mode 2  : per channel 2^bits[c] fp32 palette      (QAT_GRID[bits][0..levels], L1705-1710)
        mode 3  : none (the 4-bit scale codes are payload)
```

`H` is whatever the sum comes to; the writer records it. After the records both readers check `nin` against the channel sum plus the positional inputs and the minimum section size against `file_bytes - H` (amendment item 4). Level 0 must satisfy the v13 loader's invariants when mode 3 (nearest, full resolution, W and H multiples of 8: main.cpp L2668, ntc_decode L161) and mode 2 (nearest: L2533).

### 1.2 Payload: sections, each with an 8-byte section header

```
1  kind      (1 = level payload, 2 = mlp weights)
1  coding    (0 = raw fixed-width, the only value in v1: for a level the bit fields below, for the mlp raw fp16; see section 5 for 1, 2)
1  level     (kind 1: level index; kind 2: 0)
1  reserved  (0)
4  byte_len  (length of the section body after this header, padding included)
```
Sections appear in order: level 0, level 1, ..., mlp. The reader knows every body length from the header, computes the expected `byte_len` per section and refuses a mismatch.

Every level body opens with the 6-bit marker `NTCB_SYNC_LEVEL` and closes (before the byte padding) with `NTCB_SYNC_END`; the MLP body opens with `NTCB_SYNC_MLP` and closes with `NTCB_SYNC_END` (amendment item 3 of 1.1). The bit counts below are the content bits, markers excluded.

**Level body, mode 0 / 1 / 2:** for c = 0..C-1, for texels in (y, x) raster order, `bits[c]` bits = the grid index k. Bits = `sum_c W*H*bits[c]`. Then pad to a byte.

**Level body, mode 3 (dct), C channels, blocks BW x BH = (img_w/8) x (img_h/8):** for c = 0..C-1, the marker `NTCB_SYNC_CHANNEL`, then:
- `code[b][c]`, 4 bits each, blocks in raster order: `4 * nblk` bits (the simulator's `S.code_raw += 4.0 * nb`, L985);
- then for each block b in raster order:
  - DC: `dc_raw_bits` bits, **unsigned**, value `sym[0]`. `dc_raw_bits = ceil(log2(floor(512 / dc_step) + 1))` exactly as computed at L2573 and ntc_decode L432 (8 at step 4, 10 at step 1, 4 at step 64). The DC of a [0,64] plane is non-negative by construction (`dct_fwd8` L707-713 sums non-negative products for u = v = 0, `dct_quant_dc` L724 rounds it), so an unsigned field of that width is exactly what the simulator charges (`S.raw[0] += Q.dc_raw_bits`, L962); the writer refuses `sym[0] < 0 || sym[0] >= 2^dc_raw_bits` ("DC symbol out of range: writer bug");
  - then the 63 ACs in zigzag order (`DCT_ZIGZAG`, L624-638; `sym[DCT_ZIGZAG[p]]`, L931) as tokens: for every nonzero, `run` 7 bits (zeros since the previous nonzero, 0..62), `|q| - 1` 8 bits (0..255), `sign` 1 bit (1 = negative) = 16 bits (`S.raw[1] += 7; S.raw[2] += 8; S.raw[3] += 1`, L970); after the last nonzero, if its zigzag index `lnz < 63`, one EOB token = the run field with value 64, 7 bits (`if (lnz < DCT_NN - 1) ... S.raw[4] += 7`, L975-977). An all-zero block is DC + EOB. The writer refuses `|q| > 256` (the simulator folds those into the top bin and counts `nmag_over`, L933/L967, which must stay 0).
- Pad to a byte at the end of the section (after the last channel).

Bits of a dct level body = `sum_c [4*nblk + sum_b (dc_raw_bits + 16*nnz_b + 7*[lnz_b < 63])]` = `S.raw_total - S.header` of `dct_analyze` (L988-990 with `S.raw[6] = header`), i.e. exactly the trainer's `[dct: raw ...]` figure (L3221) and ntc_decode's `raw_bits` (L1106).

**MLP body (coding 0 = raw fp16):** `nparams` x 16 bits, fp16 (IEEE binary16, LE) in `D.mlp.p` order (per layer: `nh x ncur` weights row-major then `nh` biases, L546-552, output layer L556; ntc_decode `pack_mlp` reads the same order). Bits = `16 * nparams` = `s.bits_mlp` (L1925). Weights are already fp16-representable when the file is written (decision 2), so `f16_to_f32(f16)` reproduces `D.mlp.p` bit for bit. The writer refuses non-finite or |w| > 65504.

### 1.3 Dequantization rules the readers must implement (verbatim, strict FP)

- mode 0 / 1: `range = max(hi - lo, 1e-6f)` (L405/L415, ntc_decode L321); `value = lo + k / (float)levels * range`, `levels = (1 << bits) - 1` (`qes_rebuild` L416; ntc_decode `q8_quantize` L325, same expression). Both are inside strict-FP blocks (main.cpp L390-432; ntc_decode L210-506) so writer-side `qes_rebuild` and both readers give the same bit patterns. Writer-side index: mode 1: `qes_index(z, lo, range, levels)` (L392-397, the same call `bitrate_stats` makes at L1962) with an assertion `grid[k] == zdec[i]`; mode 0: `q8_index(z, lo, range, levels)` = `lround((z - lo)/range*levels)` clamped, with lo/hi = the per-channel min/max of the shadow (L1955), and the assertion `q8_value(k, lo, range, levels) == zpost[i]` against the dequantized latent behind `recon_q_final.png` (amendment of September 7, 2026: `bitrate_stats` computes its post-hoc index and value through the same strict helpers, because under `/fp:fast` the inline expressions gave values an ulp off the strict ones and the writer refused the first mode-0 level it met). A mode-0 grid reinstalled from a container (`QesLevel::from_ntcb`) is written as mode 0 again with its stored lo/hi and `qes_index`.
- mode 2: `value = palette[c][k]`, k = `qat_index(z, bits)` (L1711-1715), assertion `palette[k] == z`.
- mode 3: `dct_recon_level0` (main.cpp L796; ntc_decode L425) on the parsed symbols and codes: dequantize (`dct_dequant_block` L746-748), `dct_inv8` (L714-720 / ntc_decode L404, zero-skipping, bit-exact), `clamp(w/32 - 1)` (L760-761). Nothing new.

### 1.4 Expected total bits and the reconciliation with `bitrate_stats`

`bitrate_stats` (L1920-1984): `bpp_q8 = (raw_bits + header_bits + bits_mlp) / npix` (L1980), with per level: dct: `raw_bits += raw_total - header; header_bits += 96*C` (L1943-1945); qat (`fixed`): `raw_bits += ntex * bits` per channel, no header (L1953, L1978); qes: `raw_bits += ntex * bits` per channel + `64 * C` header (L1953, L1978); continuous: `raw_bits += nl * lbits` + `64 * C` (L1949, L1978).

Define `content_bits` = the sum of the unpadded section bodies (level bodies + mlp body). By construction (1.2):

```
content_bits == raw_bits + bits_mlp == sim_total_bits - sim_side_bits
sim_total_bits = round(bs.bpp_q8 * npix),  sim_side_bits = header_bits (64/channel for mode 0/1 levels, 96/channel for dct)
file_bits = 8*file_bytes = 8*H + 64*nsections + sync_bits + content_bits + pad_bits
```

The writer prints all terms and declares `OK` only when `content_bits == sim_total_bits - sim_side_bits` exactly (no tolerance); anything else is `MISMATCH` and exits 1 (a simulator bug or a writer bug, both of which this file exists to catch). To read `raw_bits` and `header_bits` the struct gains two fields, `bits_raw_latent` and `bits_header`, set at L1980 (one tagged line each; nothing else reads them). The container's own overhead relative to the simulator is then explicit: `8*H + 64*nsections + sync_bits + pad_bits - sim_side_bits` (the lo/hi floats are exactly the 64-bit charge; the dct side info is 1 + C bytes instead of 96 bits per channel; the qat palette, the fixed header, the section headers and the sync markers are extra). After building the bytes the writer re-reads them through `ntcb_read_header` + `ntcb_restore` into a copy of the decoder state and compares the recovered symbols, codes, grid values, ranges, palettes and weights with what it wrote (`self-check OK` on the file: line; any difference is a refusal).

Worked expectation for `out_image3_b8_dct50_c4bilin_qes8_mlp27_cuda8k_fd50` (its log's final block): dct `raw ... hdr 96 total 3450566` bits -> level-0 body 3450470 bits; level 1 185x178x4 at 8 bits = 1053760 bits; mlp 1056 x 16 = 16896; content = 4521126 bits; simulator raw 2.145 bpp x 1480x1424 = 4520838 +- rounding of the printed bpp, minus side info 96 + 256 = 352 -> the identity is checked on the unrounded double, not the printed 3 decimals. File ~ 565.2 KB; the header ~ 100 bytes.

## 2. Code: where it goes

### 2.1 main.cpp, new region `// ---- NTCB: bit-packed container [NTCB]` ... `// ---- end of the NTCB region [NTCB]` (~330 lines), inserted after `bitrate_stats` (L1984) and before the `eval_filter` comment (L1986), because the writer needs `Decoder`, `DctLevel`, `dct_analyze`, `qat_index`/`qat_value` (L1710-1715), `qes_index`/`qes_rebuild` (L392-418) and `BitrateStats`, all defined above that point.

- `BitWriter` (accumulator, `put(uint32 v, int n)`, `align()`, `bits()`, byte vector) and `BitReader` (bounds-checked, `ok` flag like ntc_decode's `Reader` L96-102): ~40 lines.
- `f32_to_f16(float)` (RNE, denormals, overflow refused) and `f16_to_f32(uint16)`: ~30 lines; `mlp_round_fp16(std::vector<float>&)` returning (changed, max |dw|); a startup self-check that all 65536 half patterns round-trip through `f16_to_f32`/`f32_to_f16` (run once when the flag is on).
- `struct NtcbHeader` mirroring 1.1, `ntcb_write_header`, `ntcb_read_header` (validation of every range: magic, version == 1, H <= size, file_bytes == size, hash, T 1..4, nlevels 1..3, modes, bits, q, dc_step, dims >= 1, C <= MAXH, poslen < 4096, nhidden <= MAXL, widths <= MAXH, nin == channels + pos count, nout == 3T): ~120 lines.
- `ntcb_write(const std::string& path, const Decoder& D, const BitrateStats& bs, const Options& o, NtcbReport& r)`: chooses each level's mode from `D.dct.live` (level 0), `D.qat_ch` (level 0), `D.qes_live && D.qes[l].fitted()`, else mode 0 with `lbits` as L1935; emits sections; computes `content_bits`, `pad_bits`; refuses (`return false` with a message) when `o.dct_q > 0 && !D.dct.live` ("level 0 is not DCT-coded yet: nothing to write; raise --dct-start or --iters"), on `nmag_over`, DC out of range, non-fp16 weight: ~110 lines. The DC/AC tokenization loop is the pass-2 loop of `dct_analyze` (L956-978) minus the entropy terms; keep it side by side with the same variable names so the two stay in step.
- `ntcb_restore(Decoder& D, const NtcbHeader& h, BitReader& r, std::vector<int>& saved_qes, ...)`: fills `D.lat.z` (modes 0/1/2), `D.dct.sym`/`D.dct.code` (mode 3), `D.mlp.p`; returns the per-level lo/hi and bits so the loader can install grids: ~80 lines.

### 2.2 main.cpp hooks (each line tagged `[NTCB]`)

- `Options` (after L165 `resave`): `bool write_ntcb = false;`. Parser (next to `--resave`, L2387): `--write-ntcb`. `usage()` (near L238): two lines.
- `BitrateStats` (L1881-1888): `double bits_raw_latent, bits_header;` set after L1979.
- Loader (L2592-2762): at L2602 the first six ints are read; add before L2615: `const bool is_ntcb = hdr[0] == NTCB_MAGIC;` If so: rewind, `ntcb_read_header`, and fill the same `saved_*` locals (`saved_lv`, `saved_filter`, `saved_hidden`, `saved_pos`, `saved_T`, `saved_leak`, `saved_qat`/`saved_qat_ch` (mode 2 bits), `saved_qes` (mode 1 bits) + `saved_qes_lo/hi`, `saved_dct_N/dc/q` (mode 3)), skipping the fread cascade (`if (!is_ntcb) { existing L2615-2687 } else { fill }`), so the mismatch check at L2703-2723 and its message run unchanged for both formats. Payload: at L2724 `if (is_ntcb) ok = ntcb_restore(...) else if (saved_dct_N > 0) ... else ...`. Post-load steps then work as they are: qat snap L2738-2741 (moved = 0 because the palette is `QAT_GRID`), qes restore L2744-2749 (values are grid values; moved = 0), dct reconstruction L2754-2759 (from the parsed symbols). For **mode-0 levels** add one block after L2753: install the file's lo/hi as a frozen `QesLevel` with `bits = B` (`qes_rebuild`), `D.zq = D.lat.z; D.qes_live = D.qes_frozen = true;` and print `ntcb     : level l: 8-bit grid restored from the file (frozen; its stored values are the run's post-hoc quantization)`. Reason: `bitrate_stats` would otherwise re-quantize the dequantized values with `hi' = lo + range`, which can differ from `hi` by an ulp (L1955-1966); through the qes branch (L1953, L1962, L1978) the raw / header charges are identical to the continuous branch's and the values are returned untouched, so (c) is exact for every mode. Consequence to state: continuing training (`--iters > 0`) from such a file treats those levels as `--qes B` levels; print a note and leave it allowed.
- Loop (L3092-3096): after the device download at L3094 and before `decode_with` at L3096: `if (o.write_ntcb && it == o.iters) { round; print }`. After the loop (before L3163): `if (o.write_ntcb && o.iters == 0) { round; print }`.
- Final block (L3167-3224): after `bs` is computed at L3168 and the PNGs are written (L3195-3207), before `done:` (L3208): `if (o.write_ntcb) { NtcbReport r; if (!ntcb_write(o.outdir + "/model.ntcb", D, bs, o, r)) return 1; print the file: line }`. Print `file: <path> <bytes> bytes = <bpp> bpp (simulator raw <bpp> bpp); payload <content_bits> bits = simulator <sim_total> - <side> side-info bits [OK|MISMATCH]; container header <8H> + section headers <64n> + padding <pad> bits`. `done:` stays the last line; a MISMATCH sets the exit code to 1 after `done:`.
- `--resave` (L3163) is untouched; `model.bin` then holds the rounded weights (decision 2).

### 2.3 ntc_decode.cpp

- `Model` (L74-94): `bool ntcb = false; int ntcb_version = 0; std::vector<int> ntcb_mode; std::vector<int> ntcb_bits;` (~4 lines).
- `load_model` (L104-185): after `int magic = r.i32();` at L110: `if (magic == NTCB_MAGIC) return load_ntcb(buf, m, err);` (before the `0x4E5443` test at L111).
- New region after the DCT region end (L460), still inside the strict-FP block that starts at L210 (the mode 0/1 dequantization must be strict, like `q8_quantize` at L311-329): `BitReader`, `f16_to_f32`, `load_ntcb(buf, m, err)` (~180 lines): parses 1.1, sets `m.version = 13` (so the v11+ size/crop/clamp branches at L991, L996, L999 apply) and `m.ntcb = true`, `m.imgW/H`, `m.srcW/H`, `m.clamp`, `m.leak`, `m.hidden`, `m.pos`, `m.T`, `m.nin`, `m.lv`; per level: mode 2 -> `m.qat_ch` and values from the palette; mode 0/1 -> values from the grid, `m.qes_bits[l] = B`, `m.qes_lo/hi` (so the existing `level_quantized` L309 returns true and `q8_quantize` L314 skips it, and the summary L1096-1100 prints the range); mode 3 -> `m.dct_N/dc_step/C/q/sym/code` (so the existing L1011-1015 reconstruction runs); mlp from fp16; section lengths, hash, file length, trailing bytes all checked with clear `err` strings.
- Summary: L1083 prints `(v%d)`; hook: `m.ntcb ? "ntcb v1" : "v13"`. Level line L1088-1101: mode 0 levels show as `qes 8-bit, range ...` through the qes branch; add the word `ntcb q8` via `m.ntcb_mode` (one hook in that branch). New `file     :` line after the `dct`/`idct` lines (L1104-1112): `file     : model.ntcb <bytes> bytes = <bpp> bpp (simulator raw <bpp>: level bits + <side> side-info + <mlp> mlp bits; payload matches the simulator [exactly | MISMATCH])`, where ntc_decode computes the simulator figure itself from its header and `DctReconStats` (`raw_bits` at L1106 is already `raw_total - header`) plus `64 * C` per mode-0/1 level, `96 * C` for dct, `16 * nparams`. The `--pack-selectors` note (L1054) and everything else unchanged.

Estimated new code: main.cpp ~370 lines (region 330 + hooks 40), ntc_decode.cpp ~200, README ~25, NTCB_NOTES.md. Nothing in `cuda/`.

## 3. Round-trip verification (commands; every figure to be quoted from program output in NTCB_NOTES.md)

(a) **Size vs simulator.** `ntc image3.png --cuda --block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 50 --qes 0,8 --mlp 27,27 --leak 0.0009765625 --load out_image3_b8_dct50_c4bilin_qes8_mlp27_cuda8k_fd50/model.bin --iters 0 --write-ntcb --resave --out out_ntcb_dct50` (the v13 load path, L2754-2759; `--resave` rewrites model.bin with the rounded weights). Expect the `file:` line to print `OK` with `content_bits == sim_total - 352` and the file bpp within `(8H + 64n + pad - 352)/npix` of the done-line raw bpp. Also on the CPU-trained `out_dct_a_cpu` and the 5-iteration `out_chk` (DCT_NOTES section 3) so an EOB-heavy plane (67% EOB-only) is covered.

(b) **Decoder byte identity.** `ntc_decode out_ntcb_dct50/model.ntcb -o n --compare out_ntcb_dct50/recon_q_final.png --verify` and `ntc_decode out_ntcb_dct50/model.bin -o b --compare ... --verify`: `cmp n.png b.png` identical; `compare t0` max |diff| <= 1 and PSNR > 85 dB against `recon_q_final.png` (the trainer's PNG is written from the same weights and plane; the 1-LSB tolerance is the existing fast-sigmoid tolerance, CPU_DECODER_NOTES section 3); `verify` 0 ulp pre-nonlinearity. Also `--fp32-latent` vs `--q8`: identical (every level is quantized).

(c) **Trainer reload.** `ntc image3.png <same options> --load out_ntcb_dct50/model.ntcb --iters 0 --out out_ntcb_dct50_reload`: `iter 0 psnr` and the `done:` line equal to the writing invocation's `done:` line (PSNR, raw / entropy / ctx bpp, the `[dct: ...]` bracket) on the CPU and with `--cuda`. Loading it without `--dct-q`, with `--dct-q 30`, with `--mlp 17,17`: the existing "was saved with ... pass the same options" message (L2719), rc 1.

(d) **qat-mode file.** `ntc image3.png --cuda --block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --qat 4 --qes 0,8 --mlp 27,27 --leak 0.0009765625 --load out_image3_b8_c1q4_c4bilin_qes8_mlp27_cuda8k_fd50_psnr8/model.bin --iters 0 --write-ntcb --resave --out out_ntcb_q4`: `file:` OK (level 0: 1480x1424x4 bits; palette 16 floats; level 1 qes 8), (b) and (c) as above (the PSNR is the converting invocation's, i.e. after the fp16 rounding; quote the delta against the log's 48.94 dB). Level-0 section bits must equal `W*H*4`.

(e) **4-texture material.** `ntc m1.png m2.png m3.png m4.png --cuda --latent 512 512 2 --latent2 128 128 4 --filter nearest,nearest --pos lv1local --qat 2 --mlp 36,36 --leak 0.0009765625 --load out_m1234_512c2q2_128c4_cuda8k_fd50/model.bin --iters 0 --write-ntcb --resave --out out_ntcb_m1234`: qat 2,2 level 0 (two channels, two palettes), a mode-0 level 1 (exercises the frozen-grid install of 2.2), T = 4 (four `recon_q_final_tK.png`, `ntc_decode --compare a,b,c,d`, four PNG byte comparisons). Also `out_m1234` (bilinear 128x128x4 level 0, mode 0 on level 0, `--pos uv`, 3000 iterations, v12): mode-0 level 0 with a bilinear filter.

(f) **Refusals.** A script (PowerShell in the scratchpad) that produces: the file truncated at 20, at H - 1, at H + 100, at file_bytes - 1 bytes; one byte flipped in the payload (hash), in a level record (mode 9, bits 13, q 0, dc_step 0, filter 2), in the section header (byte_len +1), magic changed, version 2; a hand-edited token stream with run 63 (not EOB and beyond position 63) and with more than 63 AC positions in a block. Each: `ntc_decode` prints the message, rc 1, no crash; `ntc --load` prints "corrupt or truncated" / the specific message, rc 1. Also `--write-ntcb` on a `--dct-q` run whose switch never happened (`--dct-start 1.0 --iters 5`): the refusal, rc 1.

(g) **Firewall.** With the flag absent no new code runs: regression pin `ntc --crop 512 --mlp-pairs 32 --iters 200 --print-every 200 --save-every 200 --out out_reg` prints `psnr 23.83 dB` and `out_reg/model.bin` has the sha256 of DCT_NOTES section 3 (`cb884707...`); the six README `--cuda-check` lines (plus the `--dct-q 90` / `20` variants) unchanged; `ntc_decode --compare --verify` on the v10 / v11 / v12 / v13 files of DCT_NOTES section 3 prints the same figures. Build with 0 warnings, CPU and CUDA.

## 4. Reporting

- Trainer: the `file:` line of 2.2 printed once before `done:`; with T > 1 also `(/tex)` figures like the done line (L3217). The `ntcb     :` rounding line at the point of rounding.
- ntc_decode: the `file     :` line of 2.3 in the summary block; the `model    :` line says `(ntcb v1)`.
- Both print the file size in bytes and the derived bpp, next to the simulator's raw bpp, and the exact-match verdict.

## 5. The later entropy-coding drop-in

The `coding` byte of each section is the versioning point; the header and the other sections do not change. Planned values: `1` = separated streams, each zstd-compressed (XUASTC-style); `2` = adaptive arithmetic (order-0 / the simulator's `ctx` model). For a dct level under coding 1 or 2 the body becomes a small stream table followed by the streams, per channel: DC residuals (against the left block, first column above, first block 0: the predictor of `dct_analyze` L925), runs (65-symbol alphabet, EOB included), magnitudes (`|q|`, 257 symbols with the top bin), signs (raw bits, never coded: L971 charges 1 bit), scale codes (16 symbols; context = up/left for coding 2, matching `context_entropy_bits` L985). For mode 0/1/2 levels: one index stream per channel. The MLP section stays raw. The simulator's `h0` / `ctx` columns are then the targets for coding 1 / 2 the same way the `raw` column is the target for coding 0, so the reconciliation line generalizes (`file bits vs simulator <column>`). A v1 reader refuses any `coding != 0` with "coded sections need a newer decoder", so old binaries fail cleanly rather than misparse. Multi-channel level 0 (`C` up to 4) and 1..4 textures need no format change; a DCT-coded level 1 later would be mode 3 on a level >= 1 (the level record already carries the fields).

## 6. Firewall / removal, order, risks

**Firewall.** `o.write_ntcb == false` -> no rounding, no write, no `file:` line; the loader takes the ntcb branch only on the magic; `BitrateStats` gains two fields nobody reads; ntc_decode's `load_ntcb` runs only on the magic. No RNG draws, no change to any decode.

**Removal list.** main.cpp: the `Options` field, the parser line, the two usage lines, the two `BitrateStats` fields and their assignment, the `[NTCB]` region, the loader's `is_ntcb` branch (restore L2615-2687 verbatim) and the payload/`mode-0 install` hooks, the two rounding hooks, the final-block write/print. ntc_decode.cpp: the `Model` fields, the magic dispatch line, the region, the summary hooks. README rows; NTCB_NOTES.md. `grep -n "\[NTCB\]"` finds every line, as with `[DCT]`.

**Implementation order** (build and check after each):
1. Bit writer/reader + fp16 + the header struct with write/read and the self-checks (region only, no hooks). Check: a unit test in the region run under a hidden `--ntcb-selftest`, or simply the round-trip of the header on step 3.
2. `ntcb_write` + the rounding hooks + `BitrateStats` fields + the `file:` line. Check: (a) on `out_chk` and the dct50 run; the identity prints OK; hand-check one block's tokens against `dct_analyze`'s counts as DCT_NOTES did (`4096 x 8 + 229859 x 16 + 707 x 7 + 4096 x 4`).
3. ntc_decode `load_ntcb` + hooks. Check: (b), (d), (e), the `file` line, `--verify`.
4. main.cpp loader branch + mode-0 install. Check: (c), the refusals of (c).
5. (f) corruption script; (g) regression; README + NTCB_NOTES.md.

**Risks.**
- Bit widths vs the simulator: the writer must refuse rather than fold (`|q| > 256`, DC out of range); at `q 100` blocks have up to 63 nonzeros with magnitudes bounded by 256 (DCT_MVP_PLAN 13.10), verified by the `nmag_over` counter staying 0 in every log so far. Any drift between the token loop and `dct_analyze` shows up as `MISMATCH` immediately, which is the point.
- The fp16 rounding changes the run's final PSNR by a small amount and `model.bin` by the rounded weights; only under the flag, printed, and it is what the quoted 16-bit MLP charge has always implied.
- qat palette and `QAT_GRID`: the loader's `qat_snap_level0` (L2739) re-snaps through `QAT_GRID`; a build whose `/fp:fast` grid differs by an ulp would print `moved > 0` and change values (also true of `model.bin` today). The file makes it visible.
- Mode-0 exactness depends on the frozen-grid install; without it, (c) holds only to display precision on runs with a post-hoc level (see 2.2).
- Endianness: no host-order reads or writes anywhere (byte-wise packing), so the Linux/GCC `ntc_decode` build of CPU_DECODER_NOTES section 7 should produce the same PNG; worth one run.

## 7. Settings-independence matrix (added September 7, 2026)

Richard's requirement: the container must work independent of the run's settings (DCT or not, 2 or 3 levels, any quantization depth, 1..4 textures, any cell size, nearest or bilinear level 0), and that has to be tested, not assumed. Every row is a `--load <dir>/model.bin --iters 0 --write-ntcb --resave --out out_ntcb_<row>` conversion (options from the row's log), followed by the (b) decoder byte identity and (c) trainer reload checks of section 3. The `file:` verdict must be `OK` on every row.

| row | existing dir | what it covers |
|---|---|---|
| dct50 | out_image3_b8_dct50_c4bilin_qes8_mlp27_cuda8k_fd50 | dct level 0 (v13), 8x8 cells, qes 8 level 1 |
| dct-b6 | out_model11_b6_dct15_c4bilin_qes8_mlp27_refit500_cuda8k_fd50 | dct with 6x6 cells (lcm padding to 24), refits |
| dct-b4 | out_m1_b4_dct30_c4bilin_qes8_mlp27_refit500_cuda8k_fd50 | dct with 4x4 cells, 512x512 |
| dct-eob | out_chk, out_dct_a_cpu | EOB-heavy small planes, CPU-trained |
| qat4 | out_image3_b8_c1q4_c4bilin_qes8_mlp27_cuda8k_fd50_psnr8 | 4-bit searched level 0 (palette), qes 8 |
| qat1/2/3 | out_model15_b8_c1q1 / c1q2 / c1q3 _c3bilin_qes8_mlp27_cuda8k_fd50_psnr8 | 1-, 2-, 3-bit palettes, 3-channel latent |
| qat-2ch | out_model15_b4_c2q4q2_c4bilin_qes8_mlp27_cuda8k_fd50_psnr8 | two level-0 channels at different depths (4,2), 4x4 cells |
| qes6 / qes10 | out_image3_b8_c1q4_c4bilin_qes6_mlp27_cuda8k_fd50, ..._qes10_... | 6- and 10-bit block latent (index width != 8) |
| c1float | out_image3_b8_c1float_c4bilin_qes8_mlp27_cuda8k_fd50, out_model15_b8_c1float_c3bilin_qes8_mlp27_cuda8k_fd50 | continuous level 0 (mode 0, post-hoc 8-bit grid at full resolution) |
| c1qes8 | out_image3_b8_c1qes8_c4bilin_qes8_mlp27_cuda8k_fd50, ..._c6bilin_... | qes level 0 (mode 1 at full resolution), 6-channel latent |
| mat | out_m1234_512c2q2_128c4_cuda8k_fd50 | 4 textures, qat 2,2 level 0, mode-0 level 1 (frozen-grid install) |
| mat-bilin | out_m1234 (v12) | bilinear mode-0 level 0, `--pos uv` |
| mat-b6 | out_m1234_b6_c2q44_c4bilin_mlp27_cuda8k_fd50 | 4 textures, 6x6 cells, qat 4,4, post-hoc 8-bit level 1 |
| 3lv | out_model6_b6_c2q32_c4_l3c2_mlp26_cuda8k_fd50 (older format) | three levels (`--latent3`), qat 3,2, post-hoc level 1 and 2 |
| old | out_2lv_512c1_128c4_qat2 (v9), out_model12_b8_c1q2_c4bilin_mlp27_cuda8k_fd50 (v11) | loader floor and a v11 file |
| big-cells | out_model13_b32_c1q3_c4bilin_qes8_mlp27_cuda8k_fd50, out_model14_b12_c1q2_c3bilin_qes8_mlp27_cuda8k_fd50 | 32x32 and 12x12 cells |

Rows that need a fresh short run because no file exists (GPU, `--crop 512 --iters 400 --print-every 200 --save-every 200`, model.png):
- 3lv-dct: `--block 8 --latent 0 0 1 --latent2 0 0 4 --latent3 0 0 2 --filter nearest,bilinear,bilinear --pos lv1local --dct-q 50 --dct-refit 100 --qes 0,6,10 --mlp 27,27 --leak 0.0009765625` (three levels, dct level 0, 6- and 10-bit qes on levels 1 and 2).
- 3lv-qat: same with `--qat 2` instead of `--dct-q 50` and `--qes 0,8,8`.
- t2-dct: `m1.png m2.png` with `--block 4 --latent 0 0 1 --latent2 0 0 4 --dct-q 30 --qes 0,8` (two textures with a dct level 0).

The results table (row, file bytes, file bpp, simulator raw bpp, verdict, decoder max |diff| and PSNR vs recon_q_final, reload done-line equal yes/no) goes into NTCB_NOTES.md.
