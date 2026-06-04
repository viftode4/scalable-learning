#!/usr/bin/env python3
"""Render the 2D DeepSpeed Ulysses slide frames and assemble a PPTX.

The deck is intentionally image-per-slide: every build state is a separate
full-bleed 1920x1080 PNG, so the progression survives PowerPoint, Keynote, and
Google Slides import without relying on fragile generated animation metadata.

No Python package dependencies are required. Rendering needs Chrome/Chromium.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
HTML = HERE / "build.html"
FRAMES_DIR = HERE / "frames"
OUT = HERE.parent / "deepspeed-ulysses.pptx"

# Keep this list in the narrative order: research-grade 2D method walkthrough frames.
FRAMES = [
    "01_title",
    "02_memory_problem",
    "03_attention_global",
    "04_conflict",
    "05_naive_fix",
    "06_core_idea",
    "07_axes",
    "08_start_layout",
    "09_first_alltoall_packets",
    "10_after_layout",
    "11_attention_works",
    "12_values_context",
    "13_all_heads_parallel",
    "14_context_problem",
    "15_second_alltoall",
    "16_restored_layout",
    "17_layer_rhythm",
    "18_communication_intuition",
    "19_comm_formula",
    "20_scaling_story",
    "21_limits_results",
    "22_takeaway",
]

SLIDE_W_PX = 1920
SLIDE_H_PX = 1080
# 13.333333 x 7.5 inches at 914400 EMU/inch.
SLIDE_W_EMU = 12192000
SLIDE_H_EMU = 6858000

NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def find_chrome(explicit: str | None = None) -> str:
    candidates = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get("CHROME"):
        candidates.append(os.environ["CHROME"])
    candidates.extend([
        "google-chrome",
        "chromium",
        "chromium-browser",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ])
    for candidate in candidates:
        resolved = shutil.which(candidate) if not Path(candidate).exists() else candidate
        if resolved:
            return resolved
    raise SystemExit("Chrome/Chromium not found. Set CHROME=/path/to/chrome.")


def frame_png(frame: str, index: int) -> Path:
    return FRAMES_DIR / f"f{index:02d}_{frame}.png"


def render_frames(chrome: str, clean: bool = True) -> list[Path]:
    if not HTML.exists():
        raise SystemExit(f"Missing slide source: {HTML}")
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    if clean:
        for old in FRAMES_DIR.glob("f*.png"):
            old.unlink()
    outputs: list[Path] = []
    for i, frame in enumerate(FRAMES):
        out = frame_png(frame, i)
        url = f"{HTML.as_uri()}?frame={frame}"
        cmd = [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-first-run",
            "--allow-file-access-from-files",
            f"--window-size={SLIDE_W_PX},{SLIDE_H_PX}",
            "--force-device-scale-factor=1",
            "--virtual-time-budget=1200",
            f"--screenshot={out}",
            url,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not out.exists() or out.stat().st_size < 10_000:
            raise SystemExit(f"Render failed or produced tiny file: {out}")
        outputs.append(out)
        print(f"rendered {out.relative_to(HERE.parent)}")
    return outputs


def rels_xml(rels: list[tuple[str, str, str]]) -> str:
    body = "".join(
        f'<Relationship Id="{escape(rid)}" Type="{escape(kind)}" Target="{escape(target)}"/>'
        for rid, kind, target in rels
    )
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Relationships xmlns="{PKG_REL}">{body}</Relationships>'


def content_types(slide_count: int) -> str:
    overrides = [
        ("/docProps/core.xml", "application/vnd.openxmlformats-package.core-properties+xml"),
        ("/docProps/app.xml", "application/vnd.openxmlformats-officedocument.extended-properties+xml"),
        ("/ppt/presentation.xml", "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"),
        ("/ppt/presProps.xml", "application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"),
        ("/ppt/viewProps.xml", "application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"),
        ("/ppt/tableStyles.xml", "application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"),
        ("/ppt/theme/theme1.xml", "application/vnd.openxmlformats-officedocument.theme+xml"),
        ("/ppt/slideMasters/slideMaster1.xml", "application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"),
        ("/ppt/slideLayouts/slideLayout1.xml", "application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"),
    ]
    overrides.extend((f"/ppt/slides/slide{i}.xml", "application/vnd.openxmlformats-officedocument.presentationml.slide+xml") for i in range(1, slide_count + 1))
    override_xml = "".join(f'<Override PartName="{part}" ContentType="{ctype}"/>' for part, ctype in overrides)
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  {override_xml}
</Types>'''


def core_props() -> str:
    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>DeepSpeed Ulysses: Tokens to Heads to Attention to Tokens</dc:title>
  <dc:creator>scalable-learning</dc:creator>
  <cp:lastModifiedBy>scalable-learning</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>'''


def app_props(slide_count: int) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Python stdlib OOXML builder</Application>
  <PresentationFormat>Widescreen</PresentationFormat>
  <Slides>{slide_count}</Slides>
  <Notes>0</Notes>
  <HiddenSlides>0</HiddenSlides>
  <MMClips>0</MMClips>
  <ScaleCrop>false</ScaleCrop>
  <HeadingPairs><vt:vector size="2" baseType="variant"><vt:variant><vt:lpstr>Slides</vt:lpstr></vt:variant><vt:variant><vt:i4>{slide_count}</vt:i4></vt:variant></vt:vector></HeadingPairs>
  <TitlesOfParts><vt:vector size="{slide_count}" baseType="lpstr">{''.join(f'<vt:lpstr>Slide {i}</vt:lpstr>' for i in range(1, slide_count + 1))}</vt:vector></TitlesOfParts>
</Properties>'''


def presentation_xml(slide_count: int) -> str:
    slide_ids = "".join(f'<p:sldId id="{255 + i}" r:id="rId{i + 1}"/>' for i in range(1, slide_count + 1))
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="{NS_A}" xmlns:r="{NS_R}" xmlns:p="{NS_P}">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>{slide_ids}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W_EMU}" cy="{SLIDE_H_EMU}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle><a:defPPr><a:defRPr lang="en-US"/></a:defPPr></p:defaultTextStyle>
</p:presentation>'''


def presentation_rels(slide_count: int) -> str:
    rels = [("rId1", f"{NS_R}/slideMaster", "slideMasters/slideMaster1.xml")]
    rels.extend((f"rId{i + 1}", f"{NS_R}/slide", f"slides/slide{i}.xml") for i in range(1, slide_count + 1))
    return rels_xml(rels)


def sp_tree_header() -> str:
    return '''<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'''


def slide_master_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="{NS_A}" xmlns:r="{NS_R}" xmlns:p="{NS_P}">
  <p:cSld><p:spTree>{sp_tree_header()}</p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>'''


def slide_layout_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="{NS_A}" xmlns:r="{NS_R}" xmlns:p="{NS_P}" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree>{sp_tree_header()}</p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>'''


def theme_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="{NS_A}" name="UlyssesDark">
  <a:themeElements>
    <a:clrScheme name="UlyssesDark">
      <a:dk1><a:srgbClr val="070A12"/></a:dk1><a:lt1><a:srgbClr val="F7FBFF"/></a:lt1>
      <a:dk2><a:srgbClr val="0F172A"/></a:dk2><a:lt2><a:srgbClr val="AAB8CB"/></a:lt2>
      <a:accent1><a:srgbClr val="67E8F9"/></a:accent1><a:accent2><a:srgbClr val="7CF7B1"/></a:accent2>
      <a:accent3><a:srgbClr val="B997FF"/></a:accent3><a:accent4><a:srgbClr val="FF6F91"/></a:accent4>
      <a:accent5><a:srgbClr val="FFD166"/></a:accent5><a:accent6><a:srgbClr val="FF456B"/></a:accent6>
      <a:hlink><a:srgbClr val="67E8F9"/></a:hlink><a:folHlink><a:srgbClr val="B997FF"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Aptos"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Clean"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
  <a:objectDefaults/><a:extraClrSchemeLst/>
</a:theme>'''


def slide_xml(image_name: str) -> str:
    image_name = escape(image_name)
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="{NS_A}" xmlns:r="{NS_R}" xmlns:p="{NS_P}">
  <p:cSld><p:spTree>{sp_tree_header()}
    <p:pic>
      <p:nvPicPr><p:cNvPr id="2" name="{image_name}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>
      <p:blipFill><a:blip r:embed="rId2"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
      <p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{SLIDE_W_EMU}" cy="{SLIDE_H_EMU}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
    </p:pic>
  </p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>'''


def assemble_pptx(pngs: list[Path], out: Path) -> None:
    if not pngs:
        raise SystemExit("No PNG frames to assemble")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("[Content_Types].xml", content_types(len(pngs)))
        z.writestr("_rels/.rels", rels_xml([
            ("rId1", f"{NS_R}/officeDocument", "ppt/presentation.xml"),
            ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
            ("rId3", f"{NS_R}/extended-properties", "docProps/app.xml"),
        ]))
        z.writestr("docProps/core.xml", core_props())
        z.writestr("docProps/app.xml", app_props(len(pngs)))
        z.writestr("ppt/presentation.xml", presentation_xml(len(pngs)))
        z.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(len(pngs)))
        z.writestr("ppt/presProps.xml", f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentationPr xmlns:p="{NS_P}"/>')
        z.writestr("ppt/viewProps.xml", f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:viewPr xmlns:p="{NS_P}"/>')
        z.writestr("ppt/tableStyles.xml", f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:tblStyleLst xmlns:a="{NS_A}" def="{{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}}"/>')
        z.writestr("ppt/theme/theme1.xml", theme_xml())
        z.writestr("ppt/slideMasters/slideMaster1.xml", slide_master_xml())
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", rels_xml([
            ("rId1", f"{NS_R}/slideLayout", "../slideLayouts/slideLayout1.xml"),
            ("rId2", f"{NS_R}/theme", "../theme/theme1.xml"),
        ]))
        z.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout_xml())
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", rels_xml([
            ("rId1", f"{NS_R}/slideMaster", "../slideMasters/slideMaster1.xml"),
        ]))
        for i, png in enumerate(pngs, start=1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide_xml(png.name))
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", rels_xml([
                ("rId1", f"{NS_R}/slideLayout", "../slideLayouts/slideLayout1.xml"),
                ("rId2", f"{NS_R}/image", f"../media/image{i}.png"),
            ]))
            z.write(png, f"ppt/media/image{i}.png")
    print(f"wrote {out} · {len(pngs)} slides")


def existing_frames() -> list[Path]:
    pngs = [frame_png(frame, i) for i, frame in enumerate(FRAMES)]
    missing = [p for p in pngs if not p.exists()]
    if missing:
        raise SystemExit("Missing rendered frames: " + ", ".join(str(p) for p in missing[:5]))
    return pngs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-render", action="store_true", help="assemble from existing PNG frames")
    parser.add_argument("--render-only", action="store_true", help="render PNG frames but do not write PPTX")
    parser.add_argument("--chrome", help="path to Chrome/Chromium binary")
    parser.add_argument("--out", type=Path, default=OUT, help="output .pptx path")
    args = parser.parse_args(argv)

    if args.skip_render:
        pngs = existing_frames()
    else:
        pngs = render_frames(find_chrome(args.chrome))
    if not args.render_only:
        assemble_pptx(pngs, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
