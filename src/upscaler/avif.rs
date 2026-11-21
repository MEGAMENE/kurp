use std::io::Cursor;
use std::sync::LazyLock;

use image::metadata::Orientation;
use image::{
    DynamicImage, GrayAlphaImage, GrayImage, ImageBuffer, ImageDecoder, ImageFormat, ImageReader,
    Luma, LumaA, Rgb, RgbImage, Rgba, RgbaImage,
};
use zenavif::{DecoderConfig, ImageMirror, ImageRotation, ManagedAvifDecoder, PixelBuffer};

/// Maximum box nesting depth when scanning ISOBMFF containers (guards against malicious nesting).
const MAX_BOX_DEPTH: usize = 16;

/// Maximum pixel count allowed during AVIF decode (64 MP decompression-bomb safety cap).
const AVIF_MAX_FRAME_PIXELS: u32 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AvifColorInfo {
    pub primaries: Option<u16>,
    pub transfer: Option<u16>,
    pub matrix: Option<u16>,
    pub full_range: Option<bool>,
    pub icc_profile: Option<Vec<u8>>,
    pub is_hdr: bool,
}

/// Decoded image together with magic-byte-detected container format and preserved ICC profile.
#[derive(Debug, Clone)]
pub struct DecodedImage {
    pub image: DynamicImage,
    pub detected_format: ImageFormat,
    pub icc_profile: Option<Vec<u8>>,
}

/// Standard ICC v2.4.0-conformant Display P3 profile (D65->D50 Bradford, sRGB TRC).
static DISPLAY_P3_ICC_STORAGE: LazyLock<Vec<u8>> = LazyLock::new(|| {
    build_rgb_icc_v2(
        "Display P3",
        // Display P3 D65 -> D50 Bradford chromatic adaptation matrix
        [0.515121, 0.241196, -0.001053],
        [0.291977, 0.692245, 0.041885],
        [0.157104, 0.066559, 0.784073],
        &srgb_eotf_table_256(),
    )
});

/// Standard ICC v2.4.0-conformant Rec. 2020 SDR profile (D65->D50 Bradford, BT.2020 SDR TRC).
static REC2020_ICC_STORAGE: LazyLock<Vec<u8>> = LazyLock::new(|| {
    build_rgb_icc_v2(
        "ITU-R BT.2020",
        // Rec. 2020 D65 -> D50 Bradford chromatic adaptation matrix
        [0.673462, 0.279037, -0.001938],
        [0.165665, 0.675339, 0.029984],
        [0.125076, 0.045624, 0.796860],
        &bt2020_sdr_eotf_table_256(),
    )
});

/// Standard ICC v2.4.0-conformant Rec. 2020 with sRGB TRC (`p=9, t=13`).
static REC2020_SRGB_ICC_STORAGE: LazyLock<Vec<u8>> = LazyLock::new(|| {
    build_rgb_icc_v2(
        "ITU-R BT.2020 sRGB",
        [0.673462, 0.279037, -0.001938],
        [0.165665, 0.675339, 0.029984],
        [0.125076, 0.045624, 0.796860],
        &srgb_eotf_table_256(),
    )
});

pub fn display_p3_icc() -> &'static [u8] {
    &DISPLAY_P3_ICC_STORAGE
}

pub fn rec2020_icc() -> &'static [u8] {
    &REC2020_ICC_STORAGE
}

pub fn rec2020_srgb_icc() -> &'static [u8] {
    &REC2020_SRGB_ICC_STORAGE
}

fn srgb_eotf_table_256() -> [u16; 256] {
    let mut table = [0u16; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        let s = i as f64 / 255.0;
        let lin = if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        };
        *slot = (lin.clamp(0.0, 1.0) * 65535.0).round() as u16;
    }
    table
}

fn bt2020_sdr_eotf_table_256() -> [u16; 256] {
    // ITU-R BT.2020 / BT.709 exact inverse OETF constants
    const ALPHA: f64 = 1.09929682680944;
    const BETA: f64 = 0.018053968510807;
    const KNEE: f64 = 4.5 * BETA;

    let mut table = [0u16; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        let v = i as f64 / 255.0;
        let lin = if v < KNEE {
            v / 4.5
        } else {
            ((v + (ALPHA - 1.0)) / ALPHA).powf(1.0 / 0.45)
        };
        *slot = (lin.clamp(0.0, 1.0) * 65535.0).round() as u16;
    }
    table
}

#[inline]
fn to_s15fixed16(v: f64) -> [u8; 4] {
    let fixed = (v * 65536.0).round() as i32;
    fixed.to_be_bytes()
}

fn push_padded_tag(data: &mut Vec<u8>, tag_bytes: &[u8]) -> (u32, u32) {
    let offset = data.len() as u32;
    let size = tag_bytes.len() as u32;
    data.extend_from_slice(tag_bytes);
    while !data.len().is_multiple_of(4) {
        data.push(0);
    }
    (offset, size)
}

fn build_xyz_tag(xyz: [f64; 3]) -> Vec<u8> {
    let mut out = Vec::with_capacity(20);
    out.extend_from_slice(b"XYZ ");
    out.extend_from_slice(&0u32.to_be_bytes());
    out.extend_from_slice(&to_s15fixed16(xyz[0]));
    out.extend_from_slice(&to_s15fixed16(xyz[1]));
    out.extend_from_slice(&to_s15fixed16(xyz[2]));
    out
}

fn build_desc_tag(description: &str) -> Vec<u8> {
    // ICC.1:2001-04 § 6.5.17 textDescriptionType:
    // 4 ('desc') + 4 (reserved) + 4 (ascii count incl. NUL) + ascii + NUL
    // + 4 (unicode lang) + 4 (unicode count = 0) + 2 (scriptcode) + 1 (mac count = 0) + 67 (mac buf)
    let ascii = description.as_bytes();
    let ascii_count = (ascii.len() + 1) as u32;
    let mut out = Vec::with_capacity(90 + ascii.len() + 1);
    out.extend_from_slice(b"desc");
    out.extend_from_slice(&0u32.to_be_bytes());
    out.extend_from_slice(&ascii_count.to_be_bytes());
    out.extend_from_slice(ascii);
    out.push(0); // NUL terminator
    out.extend_from_slice(&0u32.to_be_bytes()); // Unicode language code
    out.extend_from_slice(&0u32.to_be_bytes()); // Unicode count = 0
    out.extend_from_slice(&0u16.to_be_bytes()); // ScriptCode code
    out.push(0); // Macintosh description count = 0
    out.resize(out.len() + 67, 0); // Fixed 67-byte Macintosh description buffer
    out
}

fn build_cprt_tag(text: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + text.len() + 1);
    out.extend_from_slice(b"text");
    out.extend_from_slice(&0u32.to_be_bytes());
    out.extend_from_slice(text.as_bytes());
    out.push(0);
    out
}

fn build_curv_tag(lut: &[u16; 256]) -> Vec<u8> {
    let mut out = Vec::with_capacity(12 + lut.len() * 2);
    out.extend_from_slice(b"curv");
    out.extend_from_slice(&0u32.to_be_bytes());
    out.extend_from_slice(&(lut.len() as u32).to_be_bytes());
    for &entry in lut {
        out.extend_from_slice(&entry.to_be_bytes());
    }
    out
}

fn build_rgb_icc_v2(
    desc: &str,
    r_xyz: [f64; 3],
    g_xyz: [f64; 3],
    b_xyz: [f64; 3],
    trc_lut: &[u16; 256],
) -> Vec<u8> {
    const TAG_COUNT: u32 = 9;
    let header_and_table_len = 128 + 4 + (TAG_COUNT as usize) * 12;

    let mut payload = vec![0u8; header_and_table_len];

    let desc_loc = push_padded_tag(&mut payload, &build_desc_tag(desc));
    let cprt_loc = push_padded_tag(&mut payload, &build_cprt_tag("CC0"));
    // Media white point: D65
    let wtpt_loc = push_padded_tag(&mut payload, &build_xyz_tag([0.9504559, 1.0, 1.0890578]));
    let rxyz_loc = push_padded_tag(&mut payload, &build_xyz_tag(r_xyz));
    let gxyz_loc = push_padded_tag(&mut payload, &build_xyz_tag(g_xyz));
    let bxyz_loc = push_padded_tag(&mut payload, &build_xyz_tag(b_xyz));
    let trc_loc = push_padded_tag(&mut payload, &build_curv_tag(trc_lut));

    let total_size = payload.len() as u32;

    // 128-byte ICC v2.1.0 header
    payload[0..4].copy_from_slice(&total_size.to_be_bytes());
    payload[4..8].copy_from_slice(b"kurp"); // Preferred CMM
    payload[8..12].copy_from_slice(&[0x02, 0x10, 0x00, 0x00]); // Profile version 2.1.0
    payload[12..16].copy_from_slice(b"mntr"); // Display device class
    payload[16..20].copy_from_slice(b"RGB "); // Data color space
    payload[20..24].copy_from_slice(b"XYZ "); // PCS
    // Creation timestamp: 2026-01-01 00:00:00
    payload[24..26].copy_from_slice(&2026u16.to_be_bytes());
    payload[26..28].copy_from_slice(&1u16.to_be_bytes());
    payload[28..30].copy_from_slice(&1u16.to_be_bytes());
    payload[36..40].copy_from_slice(b"acsp"); // Profile file signature
    // PCS Illuminant: D50 (0.9642, 1.0000, 0.8249)
    payload[68..72].copy_from_slice(&0x0000_F6D6u32.to_be_bytes());
    payload[72..76].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    payload[76..80].copy_from_slice(&0x0000_D32Du32.to_be_bytes());

    // Tag table
    payload[128..132].copy_from_slice(&TAG_COUNT.to_be_bytes());
    let tags: [([u8; 4], (u32, u32)); 9] = [
        (*b"desc", desc_loc),
        (*b"cprt", cprt_loc),
        (*b"wtpt", wtpt_loc),
        (*b"rXYZ", rxyz_loc),
        (*b"gXYZ", gxyz_loc),
        (*b"bXYZ", bxyz_loc),
        (*b"rTRC", trc_loc),
        (*b"gTRC", trc_loc),
        (*b"bTRC", trc_loc),
    ];

    for (idx, (sig, (off, len))) in tags.iter().enumerate() {
        let base = 132 + idx * 12;
        payload[base..base + 4].copy_from_slice(sig);
        payload[base + 4..base + 8].copy_from_slice(&off.to_be_bytes());
        payload[base + 8..base + 12].copy_from_slice(&len.to_be_bytes());
    }

    payload
}

/// Synthesizes a conformant SDR ICC profile for wide-gamut CICP primaries ONLY when transfer
/// characteristics are SDR (never stamping SDR profiles onto HDR PQ `t=16`, HLG `t=18`, or Linear `t=8`).
pub fn synthesize_icc_from_cicp(primaries: u16, transfer: u16) -> Option<Vec<u8>> {
    // Reject HDR (PQ=16, HLG=18) and Linear (8) unconditionally
    if matches!(transfer, 8 | 16 | 18) {
        return None;
    }

    match (primaries, transfer) {
        // Display P3 (SMPTE EG 432-1) with sRGB or Unspecified TRC
        (12, 13 | 2) => Some(display_p3_icc().to_vec()),
        // Rec. 2020 with BT.709 / BT.601 / BT.2020-10/12 / Unspecified SDR TRC
        (9, 1 | 6 | 14 | 15 | 2) => Some(rec2020_icc().to_vec()),
        // Rec. 2020 with sRGB TRC
        (9, 13) => Some(rec2020_srgb_icc().to_vec()),
        _ => None,
    }
}

/// Safely reads an ISOBMFF box header at `pos` within `end`, returning `(fourcc, payload_start, box_end)`.
/// Guarantees checked arithmetic and strict forward progress (`box_end > pos`).
fn read_box_header(data: &[u8], pos: usize, end: usize) -> Option<(&[u8; 4], usize, usize)> {
    let min_end = pos.checked_add(8)?;
    if min_end > end {
        return None;
    }

    let size_u32 = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
    let fourcc: &[u8; 4] = data[pos + 4..pos + 8].try_into().ok()?;

    let (hdr_len, box_end) = if size_u32 == 1 {
        let large_end = pos.checked_add(16)?;
        if large_end > end {
            return None;
        }
        let size_u64 = u64::from_be_bytes([
            data[pos + 8],
            data[pos + 9],
            data[pos + 10],
            data[pos + 11],
            data[pos + 12],
            data[pos + 13],
            data[pos + 14],
            data[pos + 15],
        ]);
        let size = usize::try_from(size_u64).ok()?;
        if size < 16 {
            return None;
        }
        let box_end = pos.checked_add(size)?;
        if box_end > end {
            return None;
        }
        (16usize, box_end)
    } else if size_u32 == 0 {
        (8usize, end)
    } else {
        let size = size_u32 as usize;
        if size < 8 {
            return None;
        }
        let box_end = pos.checked_add(size)?;
        if box_end > end {
            return None;
        }
        (8usize, box_end)
    };

    let payload_start = pos.checked_add(hdr_len)?;
    if box_end <= pos || payload_start > box_end {
        return None;
    }

    Some((fourcc, payload_start, box_end))
}

/// Detects AVIF containers via ISOBMFF `ftyp` major/compatible brand inspection
/// (`avif`, `avis`, or `mif1`/`miaf` without non-AVIF HEIC brands).
pub fn is_avif_bytes(bytes: &[u8]) -> bool {
    let Some((fourcc, payload_start, box_end)) = read_box_header(bytes, 0, bytes.len()) else {
        return false;
    };
    if fourcc != b"ftyp" {
        return false;
    }
    let Some(major_end) = payload_start.checked_add(4) else {
        return false;
    };
    if major_end > box_end {
        return false;
    }

    let major_brand = &bytes[payload_start..major_end];
    if major_brand == b"avif" || major_brand == b"avis" {
        return true;
    }

    let mut has_avif_compat = false;
    let mut has_heic_only = false;

    // Compatible brands start after major_brand (4) + minor_version (4)
    if let Some(mut offset) = payload_start.checked_add(8) {
        while let Some(next) = offset.checked_add(4) {
            if next > box_end {
                break;
            }
            let brand = &bytes[offset..next];
            if brand == b"avif" || brand == b"avis" {
                has_avif_compat = true;
            } else if matches!(brand, b"heic" | b"heix" | b"hevc" | b"hevx" | b"msf1") {
                has_heic_only = true;
            }
            offset = next;
        }
    }

    if has_avif_compat {
        return true;
    }

    (major_brand == b"mif1" || major_brand == b"miaf") && !has_heic_only
}

#[derive(Debug, Clone)]
enum ParsedColr {
    Nclx {
        primaries: u16,
        transfer: u16,
        matrix: u16,
        full_range: bool,
    },
    Icc(Vec<u8>),
}

fn parse_colr_payload(payload: &[u8]) -> Option<ParsedColr> {
    if payload.len() < 4 {
        return None;
    }
    let color_type = &payload[0..4];
    if color_type == b"nclx" && payload.len() >= 11 {
        let primaries = u16::from_be_bytes([payload[4], payload[5]]);
        let transfer = u16::from_be_bytes([payload[6], payload[7]]);
        let matrix = u16::from_be_bytes([payload[8], payload[9]]);
        let full_range = (payload[10] & 0x80) != 0;
        Some(ParsedColr::Nclx {
            primaries,
            transfer,
            matrix,
            full_range,
        })
    } else if (color_type == b"rICC" || color_type == b"prof") && payload.len() > 4 {
        Some(ParsedColr::Icc(payload[4..].to_vec()))
    } else {
        None
    }
}

fn parse_pitm_payload(payload: &[u8]) -> Option<u32> {
    if payload.len() < 6 {
        return None;
    }
    let version = payload[0];
    if version == 0 {
        Some(u16::from_be_bytes([payload[4], payload[5]]) as u32)
    } else if payload.len() >= 8 {
        Some(u32::from_be_bytes([
            payload[4], payload[5], payload[6], payload[7],
        ]))
    } else {
        None
    }
}

fn parse_ipma_payload(payload: &[u8], target_item_id: u32) -> Vec<u16> {
    let mut associated = Vec::new();
    if payload.len() < 8 {
        return associated;
    }
    let version = payload[0];
    let flags = u32::from_be_bytes([0, payload[1], payload[2], payload[3]]);
    let wide_prop_idx = (flags & 1) != 0;
    let entry_count = u32::from_be_bytes([payload[4], payload[5], payload[6], payload[7]]);

    let mut pos = 8usize;
    for _ in 0..entry_count {
        let item_id = if version < 1 {
            let Some(next) = pos.checked_add(2) else { break };
            if next > payload.len() {
                break;
            }
            let id = u16::from_be_bytes([payload[pos], payload[pos + 1]]) as u32;
            pos = next;
            id
        } else {
            let Some(next) = pos.checked_add(4) else { break };
            if next > payload.len() {
                break;
            }
            let id = u32::from_be_bytes([
                payload[pos],
                payload[pos + 1],
                payload[pos + 2],
                payload[pos + 3],
            ]);
            pos = next;
            id
        };

        if pos >= payload.len() {
            break;
        }
        let assoc_count = payload[pos] as usize;
        pos += 1;

        for _ in 0..assoc_count {
            let prop_idx = if wide_prop_idx {
                let Some(next) = pos.checked_add(2) else { return associated };
                if next > payload.len() {
                    return associated;
                }
                let raw = u16::from_be_bytes([payload[pos], payload[pos + 1]]);
                pos = next;
                raw & 0x7FFF
            } else {
                if pos >= payload.len() {
                    return associated;
                }
                let raw = payload[pos];
                pos += 1;
                (raw & 0x7F) as u16
            };

            if item_id == target_item_id && prop_idx > 0 {
                associated.push(prop_idx);
            }
        }
    }

    associated
}

fn apply_colr_entries(entries: &[ParsedColr]) -> Option<AvifColorInfo> {
    if entries.is_empty() {
        return None;
    }
    let mut info = AvifColorInfo::default();
    let mut embedded_icc: Option<Vec<u8>> = None;

    for entry in entries {
        match entry {
            ParsedColr::Nclx {
                primaries,
                transfer,
                matrix,
                full_range,
            } => {
                info.primaries = Some(*primaries);
                info.transfer = Some(*transfer);
                info.matrix = Some(*matrix);
                info.full_range = Some(*full_range);
                info.is_hdr = matches!(*transfer, 16 | 18);
            }
            ParsedColr::Icc(icc) => {
                // Embedded rICC/prof always takes precedence over nclx synthesis
                if embedded_icc.is_none() {
                    embedded_icc = Some(icc.clone());
                }
            }
        }
    }

    if let Some(icc) = embedded_icc {
        info.icc_profile = Some(icc);
    } else if let (Some(p), Some(t)) = (info.primaries, info.transfer) {
        info.icc_profile = synthesize_icc_from_cicp(p, t);
    }

    Some(info)
}

fn parse_meta_box(data: &[u8], payload_start: usize, box_end: usize) -> Option<AvifColorInfo> {
    // FullBox: 1 byte version + 3 bytes flags
    let children_start = payload_start.checked_add(4)?;
    if children_start > box_end {
        return None;
    }

    let mut primary_item_id: Option<u32> = None;
    let mut ipco_colrs: Vec<(u16, ParsedColr)> = Vec::new();
    let mut ipma_payloads: Vec<&[u8]> = Vec::new();

    let mut pos = children_start;
    while let Some((fourcc, p_start, b_end)) = read_box_header(data, pos, box_end) {
        match fourcc {
            b"pitm" => {
                if primary_item_id.is_none() {
                    primary_item_id = parse_pitm_payload(&data[p_start..b_end]);
                }
            }
            b"iprp" => {
                let mut rp_pos = p_start;
                while let Some((rp_cc, rp_start, rp_end)) = read_box_header(data, rp_pos, b_end) {
                    match rp_cc {
                        b"ipco" => {
                            let mut prop_idx: u16 = 0;
                            let mut co_pos = rp_start;
                            while let Some((co_cc, co_start, co_end)) =
                                read_box_header(data, co_pos, rp_end)
                            {
                                prop_idx = prop_idx.saturating_add(1);
                                if co_cc == b"colr"
                                    && let Some(parsed) = parse_colr_payload(&data[co_start..co_end])
                                {
                                    ipco_colrs.push((prop_idx, parsed));
                                }
                                co_pos = co_end;
                            }
                        }
                        b"ipma" => {
                            ipma_payloads.push(&data[rp_start..rp_end]);
                        }
                        _ => {}
                    }
                    rp_pos = rp_end;
                }
            }
            _ => {}
        }
        pos = b_end;
    }

    if ipco_colrs.is_empty() {
        return None;
    }

    // If pitm + ipma associate properties with the primary item, strictly scope `colr` lookup
    // to the primary item so thumbnail/alpha/gain-map `colr` boxes cannot hijack primary color info.
    if let Some(pid) = primary_item_id
        && !ipma_payloads.is_empty()
    {
        let mut primary_props = Vec::new();
        for payload in &ipma_payloads {
            primary_props.extend(parse_ipma_payload(payload, pid));
        }
        if !primary_props.is_empty() {
            let scoped: Vec<ParsedColr> = ipco_colrs
                .iter()
                .filter(|(idx, _)| primary_props.contains(idx))
                .map(|(_, c)| c.clone())
                .collect();
            if !scoped.is_empty() {
                return apply_colr_entries(&scoped);
            }
        }
    }

    let all: Vec<ParsedColr> = ipco_colrs.into_iter().map(|(_, c)| c).collect();
    apply_colr_entries(&all)
}

/// Parses animated AVIF (`avis`) `moov -> trak -> mdia -> minf -> stbl -> stsd -> av01 -> colr`.
fn parse_moov_colr(
    data: &[u8],
    start: usize,
    end: usize,
    depth: usize,
    out: &mut Vec<ParsedColr>,
) {
    if depth >= MAX_BOX_DEPTH {
        return;
    }
    let mut pos = start;
    while let Some((fourcc, p_start, b_end)) = read_box_header(data, pos, end) {
        match fourcc {
            b"trak" | b"mdia" | b"minf" | b"stbl" => {
                parse_moov_colr(data, p_start, b_end, depth + 1, out);
            }
            b"stsd" => {
                // FullBox (4) + entry_count (4)
                if let Some(entries_start) = p_start.checked_add(8)
                    && entries_start <= b_end
                {
                    let mut e_pos = entries_start;
                    while let Some((e_cc, e_start, e_end)) = read_box_header(data, e_pos, b_end) {
                        if e_cc == b"av01" {
                            // VisualSampleEntry fixed fields = 78 bytes after box header
                            if let Some(sub_start) = e_start.checked_add(78)
                                && sub_start <= e_end
                            {
                                parse_moov_colr(data, sub_start, e_end, depth + 1, out);
                            }
                        }
                        e_pos = e_end;
                    }
                }
            }
            b"colr" => {
                if let Some(parsed) = parse_colr_payload(&data[p_start..b_end]) {
                    out.push(parsed);
                }
            }
            _ => {}
        }
        pos = b_end;
    }
}

/// Safely parses an ISOBMFF/AVIF byte buffer to extract primary color metadata (`colr` box).
///
/// - Checked arithmetic throughout (immune to 64-bit `largesize` overflow/DoS loops).
/// - Respects `pitm`/`ipma` primary-item association when present.
/// - Preserves `rICC`/`prof` embedded ICC precedence over `nclx` even when `nclx` appears first.
/// - Supports animated AVIF (`avis`) `moov/.../stsd/av01/colr`.
/// - Never stamps SDR ICC profiles onto HDR PQ (`t=16`) or HLG (`t=18`) images.
pub fn parse_avif_color_info(data: &[u8]) -> Option<AvifColorInfo> {
    if !is_avif_bytes(data) {
        return None;
    }

    let mut pos = 0usize;
    let end = data.len();
    let mut moov_colrs = Vec::new();

    while let Some((fourcc, p_start, b_end)) = read_box_header(data, pos, end) {
        match fourcc {
            b"meta" => {
                if let Some(info) = parse_meta_box(data, p_start, b_end) {
                    return Some(info);
                }
            }
            b"moov" => {
                parse_moov_colr(data, p_start, b_end, 1, &mut moov_colrs);
            }
            _ => {}
        }
        pos = b_end;
    }

    apply_colr_entries(&moov_colrs)
}

fn pixel_buffer_to_dynamic_image(pixels: PixelBuffer) -> Result<DynamicImage, String> {
    let w = pixels.width();
    let h = pixels.height();
    let desc = pixels.descriptor();
    let channels = desc.layout().channels();
    let has_alpha = desc.layout().has_alpha();
    let is_u8 = desc.channel_type().is_u8();
    let is_u16 = desc.channel_type().is_u16();
    let bpp = desc.bytes_per_pixel();

    if w == 0 || h == 0 || bpp == 0 {
        return Err(format!("invalid decoded AVIF buffer dimensions: {w}x{h}"));
    }

    let row_bytes = (w as usize)
        .checked_mul(bpp)
        .ok_or_else(|| "AVIF row byte length overflow".to_string())?;
    let total_bytes = row_bytes
        .checked_mul(h as usize)
        .ok_or_else(|| "AVIF pixel buffer size overflow".to_string())?;

    let slice = pixels.as_slice();
    let mut raw = Vec::with_capacity(total_bytes);
    for y in 0..h {
        let row = slice.row(y);
        if row.len() < row_bytes {
            return Err("AVIF decoded row shorter than expected".to_string());
        }
        raw.extend_from_slice(&row[..row_bytes]);
    }

    if is_u8 {
        match (channels, has_alpha) {
            (3, false) => RgbImage::from_raw(w, h, raw)
                .map(DynamicImage::ImageRgb8)
                .ok_or_else(|| "failed to construct Rgb8 from AVIF pixels".to_string()),
            (4, true) => RgbaImage::from_raw(w, h, raw)
                .map(DynamicImage::ImageRgba8)
                .ok_or_else(|| "failed to construct Rgba8 from AVIF pixels".to_string()),
            (1, false) => GrayImage::from_raw(w, h, raw)
                .map(DynamicImage::ImageLuma8)
                .ok_or_else(|| "failed to construct Luma8 from AVIF pixels".to_string()),
            (2, true) => GrayAlphaImage::from_raw(w, h, raw)
                .map(DynamicImage::ImageLumaA8)
                .ok_or_else(|| "failed to construct LumaA8 from AVIF pixels".to_string()),
            _ => Err(format!(
                "unsupported 8-bit AVIF layout: channels={channels}, alpha={has_alpha}"
            )),
        }
    } else if is_u16 {
        let samples: Vec<u16> = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[lo, hi]| u16::from_ne_bytes([lo, hi]))
            .collect();
        match (channels, has_alpha) {
            (3, false) => ImageBuffer::<Rgb<u16>, Vec<u16>>::from_raw(w, h, samples)
                .map(DynamicImage::ImageRgb16)
                .ok_or_else(|| "failed to construct Rgb16 from AVIF pixels".to_string()),
            (4, true) => ImageBuffer::<Rgba<u16>, Vec<u16>>::from_raw(w, h, samples)
                .map(DynamicImage::ImageRgba16)
                .ok_or_else(|| "failed to construct Rgba16 from AVIF pixels".to_string()),
            (1, false) => ImageBuffer::<Luma<u16>, Vec<u16>>::from_raw(w, h, samples)
                .map(DynamicImage::ImageLuma16)
                .ok_or_else(|| "failed to construct Luma16 from AVIF pixels".to_string()),
            (2, true) => ImageBuffer::<LumaA<u16>, Vec<u16>>::from_raw(w, h, samples)
                .map(DynamicImage::ImageLumaA16)
                .ok_or_else(|| "failed to construct LumaA16 from AVIF pixels".to_string()),
            _ => Err(format!(
                "unsupported 16-bit AVIF layout: channels={channels}, alpha={has_alpha}"
            )),
        }
    } else {
        Err(format!("unsupported AVIF channel type: {:?}", desc.channel_type()))
    }
}

/// Applies AVIF container `imir` / `irot` display transforms matching `zenavif`'s orientation mapping.
fn apply_avif_transforms(
    mut img: DynamicImage,
    rotation: Option<ImageRotation>,
    mirror: Option<ImageMirror>,
) -> DynamicImage {
    let angle = rotation.map(|r| r.angle % 360).unwrap_or(0);
    let orientation = match (mirror.map(|m| m.axis), angle) {
        (None, 0) => Orientation::NoTransforms,
        (None, 90) => Orientation::Rotate270,
        (None, 180) => Orientation::Rotate180,
        (None, 270) => Orientation::Rotate90,
        (Some(0), 0) => Orientation::FlipHorizontal,
        (Some(0), 90) => Orientation::Rotate270FlipH,
        (Some(0), 180) => Orientation::FlipVertical,
        (Some(0), 270) => Orientation::Rotate90FlipH,
        (Some(1), 0) => Orientation::FlipVertical,
        (Some(1), 90) => Orientation::Rotate90FlipH,
        (Some(1), 180) => Orientation::FlipHorizontal,
        (Some(1), 270) => Orientation::Rotate270FlipH,
        _ => Orientation::NoTransforms,
    };
    if orientation != Orientation::NoTransforms {
        img.apply_orientation(orientation);
    }
    img
}

/// Probes effective post-orientation `(width, height, detected_format)` without full pixel decode.
pub fn probe_image_dimensions(
    bytes: &[u8],
    fallback_format: Option<ImageFormat>,
) -> Result<(u32, u32, ImageFormat), String> {
    if is_avif_bytes(bytes) || fallback_format == Some(ImageFormat::Avif) {
        let cfg = DecoderConfig::new()
            .threads(1)
            .frame_size_limit(AVIF_MAX_FRAME_PIXELS);
        let decoder = ManagedAvifDecoder::new(bytes, &cfg)
            .map_err(|e| format!("failed to parse AVIF header: {e}"))?;
        let info = decoder
            .probe_info()
            .map_err(|e| format!("failed to probe AVIF dimensions: {e}"))?;
        let swap = matches!(info.rotation.map(|r| r.angle % 360), Some(90 | 270));
        let (w, h) = if swap {
            (info.height, info.width)
        } else {
            (info.width, info.height)
        };
        return Ok((w, h, ImageFormat::Avif));
    }

    let (mut decoder, detected_format) = open_image_decoder(bytes, fallback_format)?;
    let (mut w, mut h) = decoder.dimensions();
    if let Ok(orient) = decoder.orientation()
        && matches!(
            orient,
            Orientation::Rotate90
                | Orientation::Rotate270
                | Orientation::Rotate90FlipH
                | Orientation::Rotate270FlipH
        )
    {
        std::mem::swap(&mut w, &mut h);
    }
    Ok((w, h, detected_format))
}

fn open_image_decoder(
    bytes: &[u8],
    fallback_format: Option<ImageFormat>,
) -> Result<(impl ImageDecoder + '_, ImageFormat), String> {
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("failed to guess image format: {e}"))?;

    if reader.format().is_none()
        && let Some(fmt) = fallback_format
    {
        reader.set_format(fmt);
    }

    let guessed = reader.format().or(fallback_format);
    match reader.into_decoder() {
        Ok(dec) => Ok((dec, guessed.unwrap_or(ImageFormat::Png))),
        Err(first_err) => {
            if let Some(fb) = fallback_format
                && Some(fb) != guessed
            {
                let mut fallback_reader = ImageReader::new(Cursor::new(bytes));
                fallback_reader.set_format(fb);
                let dec = fallback_reader
                    .into_decoder()
                    .map_err(|e| format!("failed to decode image (fallback {fb:?}): {e}"))?;
                Ok((dec, fb))
            } else {
                Err(format!("failed to create image decoder: {first_err}"))
            }
        }
    }
}

/// Unified safe image decoder shared across ALL decode sites (`handlers/upscale.rs`,
/// `book_profile.rs`, tests).
///
/// - Detects AVIF magic bytes (`ftyp` + `avif`/`avis`/`mif1`/`miaf`) and decodes in pure safe
///   Rust via `zenavif` (applying container rotation/mirror and extracting ICC/CICP safely
///   without corrupting HDR PQ/HLG images).
/// - Decodes PNG/JPEG/WebP/GIF via `image::ImageReader`, preserving ICC profiles and applying
///   EXIF orientation.
pub fn decode_image_with_metadata(
    bytes: &[u8],
    fallback_format: Option<ImageFormat>,
) -> Result<DecodedImage, String> {
    if is_avif_bytes(bytes) || fallback_format == Some(ImageFormat::Avif) {
        let cfg = DecoderConfig::new()
            .threads(1)
            .prefer_8bit(true)
            .frame_size_limit(AVIF_MAX_FRAME_PIXELS);
        let mut decoder = ManagedAvifDecoder::new(bytes, &cfg)
            .map_err(|e| format!("failed to initialize AVIF decoder: {e}"))?;
        let (pixels, info) = decoder
            .decode_full(&zenavif::Unstoppable)
            .map_err(|e| format!("failed to decode AVIF image: {e}"))?;

        let img = pixel_buffer_to_dynamic_image(pixels)?;
        let img = apply_avif_transforms(img, info.rotation, info.mirror);

        // Extract ICC with strict pitm/ipma + rICC/prof precedence and HDR PQ/HLG safety.
        let parsed_color = parse_avif_color_info(bytes);
        let icc_profile = parsed_color
            .as_ref()
            .and_then(|c| c.icc_profile.clone())
            .or_else(|| info.icc_profile.clone())
            .or_else(|| {
                if parsed_color.is_some() {
                    // Container colr box was parsed and deliberately yielded None (e.g. HDR PQ/HLG)
                    None
                } else {
                    synthesize_icc_from_cicp(
                        u16::from(info.color_primaries.0),
                        u16::from(info.transfer_characteristics.0),
                    )
                }
            });

        return Ok(DecodedImage {
            image: img,
            detected_format: ImageFormat::Avif,
            icc_profile,
        });
    }

    let (mut decoder, detected_format) = open_image_decoder(bytes, fallback_format)?;
    let orientation = decoder.orientation().ok();
    let icc_profile = decoder.icc_profile().ok().flatten();
    let mut image = DynamicImage::from_decoder(decoder)
        .map_err(|e| format!("failed to decode image pixels: {e}"))?;

    if let Some(orient) = orientation
        && orient != Orientation::NoTransforms
    {
        image.apply_orientation(orient);
    }

    Ok(DecodedImage {
        image,
        detected_format,
        icc_profile,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_box(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let total = (8 + payload.len()) as u32;
        let mut out = Vec::with_capacity(total as usize);
        out.extend_from_slice(&total.to_be_bytes());
        out.extend_from_slice(fourcc);
        out.extend_from_slice(payload);
        out
    }

    fn make_ftyp(major: &[u8; 4], compat: &[&[u8; 4]]) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(major);
        payload.extend_from_slice(&0u32.to_be_bytes());
        for c in compat {
            payload.extend_from_slice(*c);
        }
        make_box(b"ftyp", &payload)
    }

    fn make_nclx(p: u16, t: u16, m: u16, full: bool) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(b"nclx");
        payload.extend_from_slice(&p.to_be_bytes());
        payload.extend_from_slice(&t.to_be_bytes());
        payload.extend_from_slice(&m.to_be_bytes());
        payload.push(if full { 0x80 } else { 0x00 });
        make_box(b"colr", &payload)
    }

    fn make_ricc(icc: &[u8]) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(b"rICC");
        payload.extend_from_slice(icc);
        make_box(b"colr", &payload)
    }

    #[test]
    fn test_largesize_overflow_no_hang_or_panic() {
        // 64-bit largesize = u64::MAX (previously wrapped `pos + size` to `pos - 1` -> infinite loop)
        let mut crafted = make_ftyp(b"avif", &[]);
        crafted.extend_from_slice(&1u32.to_be_bytes()); // size == 1 -> largesize
        crafted.extend_from_slice(b"meta");
        crafted.extend_from_slice(&u64::MAX.to_be_bytes());
        crafted.extend_from_slice(&[0u8; 32]);

        assert!(is_avif_bytes(&crafted));
        assert!(parse_avif_color_info(&crafted).is_none());

        // largesize < 16 (invalid)
        let mut small_large = make_ftyp(b"avif", &[]);
        small_large.extend_from_slice(&1u32.to_be_bytes());
        small_large.extend_from_slice(b"meta");
        small_large.extend_from_slice(&8u64.to_be_bytes());
        assert!(parse_avif_color_info(&small_large).is_none());
    }

    #[test]
    fn test_hdr_pq_and_hlg_never_stamped_with_sdr_icc() {
        for transfer in [16u16, 18u16, 8u16] {
            let mut avif = make_ftyp(b"avif", &[]);
            let ipco = make_box(b"ipco", &make_nclx(9, transfer, 9, true));
            let iprp = make_box(b"iprp", &ipco);
            let mut meta_payload = vec![0u8; 4]; // FullBox
            meta_payload.extend_from_slice(&iprp);
            avif.extend_from_slice(&make_box(b"meta", &meta_payload));

            let info = parse_avif_color_info(&avif).expect("should parse nclx");
            assert_eq!(info.primaries, Some(9));
            assert_eq!(info.transfer, Some(transfer));
            assert!(
                info.icc_profile.is_none(),
                "transfer={transfer} must NEVER be stamped with SDR ICC"
            );
            assert_eq!(info.is_hdr, matches!(transfer, 16 | 18));
        }

        // SDR Display P3 (p=12, t=13) DOES synthesize conformant Display P3 ICC
        let mut p3 = make_ftyp(b"avif", &[]);
        let ipco = make_box(b"ipco", &make_nclx(12, 13, 6, true));
        let iprp = make_box(b"iprp", &ipco);
        let mut meta_payload = vec![0u8; 4];
        meta_payload.extend_from_slice(&iprp);
        p3.extend_from_slice(&make_box(b"meta", &meta_payload));

        let info = parse_avif_color_info(&p3).expect("should parse P3 nclx");
        assert!(!info.is_hdr);
        assert_eq!(info.icc_profile.as_deref(), Some(display_p3_icc()));
    }

    #[test]
    fn test_ricc_precedence_over_nclx_and_pitm_ipma_scoping() {
        let custom_icc = b"CUSTOM_PRIMARY_ICC_PROFILE_DATA";
        let thumb_icc = b"THUMBNAIL_ICC_SHOULD_BE_IGNORED";

        // Property 1: thumbnail rICC (associated with item 2)
        // Property 2: primary nclx (associated with item 1)
        // Property 3: primary rICC (associated with item 1, AFTER nclx!)
        let mut ipco_payload = Vec::new();
        ipco_payload.extend_from_slice(&make_ricc(thumb_icc));
        ipco_payload.extend_from_slice(&make_nclx(12, 13, 6, true));
        ipco_payload.extend_from_slice(&make_ricc(custom_icc));
        let ipco = make_box(b"ipco", &ipco_payload);

        // ipma v0 flags=0: 2 entries: item 2 -> [prop 1], item 1 -> [prop 2, prop 3]
        let mut ipma_payload = vec![0, 0, 0, 0]; // version=0, flags=0
        ipma_payload.extend_from_slice(&2u32.to_be_bytes());
        ipma_payload.extend_from_slice(&2u16.to_be_bytes()); // item 2
        ipma_payload.push(1);
        ipma_payload.push(1); // prop 1
        ipma_payload.extend_from_slice(&1u16.to_be_bytes()); // item 1
        ipma_payload.push(2);
        ipma_payload.push(2); // prop 2 (nclx)
        ipma_payload.push(3); // prop 3 (rICC)
        let ipma = make_box(b"ipma", &ipma_payload);

        let mut iprp_payload = Vec::new();
        iprp_payload.extend_from_slice(&ipco);
        iprp_payload.extend_from_slice(&ipma);
        let iprp = make_box(b"iprp", &iprp_payload);

        // pitm v0: primary item = 1
        let mut pitm_payload = vec![0, 0, 0, 0];
        pitm_payload.extend_from_slice(&1u16.to_be_bytes());
        let pitm = make_box(b"pitm", &pitm_payload);

        let mut meta_payload = vec![0u8; 4];
        meta_payload.extend_from_slice(&pitm);
        meta_payload.extend_from_slice(&iprp);

        let mut avif = make_ftyp(b"avif", &[]);
        avif.extend_from_slice(&make_box(b"meta", &meta_payload));

        let info = parse_avif_color_info(&avif).expect("should parse scoped colr");
        assert_eq!(info.primaries, Some(12));
        assert_eq!(info.icc_profile.as_deref(), Some(custom_icc.as_slice()));
    }

    #[test]
    fn test_conformant_icc_profiles_structure() {
        for icc in [display_p3_icc(), rec2020_icc(), rec2020_srgb_icc()] {
            let declared_len = u32::from_be_bytes([icc[0], icc[1], icc[2], icc[3]]) as usize;
            assert_eq!(declared_len, icc.len());
            assert_eq!(&icc[36..40], b"acsp");
            let tag_count = u32::from_be_bytes([icc[128], icc[129], icc[130], icc[131]]);
            assert_eq!(tag_count, 9);
            // Verify desc tag size >= 95 bytes (ICC v2 spec conformance)
            let desc_size = u32::from_be_bytes([icc[140], icc[141], icc[142], icc[143]]);
            assert!(desc_size >= 95, "desc tag size {desc_size} < 95");
        }
    }
}
