"""hwp(한글 5.x, OLE)와 hwpx(zip+XML)에서 문단 글만 뽑는다. 서식·그림은 버린다."""
import re
import struct
import zipfile
import zlib
from xml.etree import ElementTree as ET

import olefile

PARA_TEXT = 67  # HWPTAG_BEGIN(16) + 51
# 8 wchar(16바이트)를 차지하는 인라인·확장 제어 문자
_WIDE = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}


def _para_text(data):
    """글자는 UTF-16LE 바이트로 모아 한 번에 디코딩한다 (서로게이트 쌍을 올바르게 합치기 위해)."""
    buf, i = bytearray(), 0
    while i + 1 < len(data):
        ch = struct.unpack_from("<H", data, i)[0]
        if ch in _WIDE:
            if ch == 9:
                buf += "\t".encode("utf-16-le")
            i += 16
            continue
        if ch in (10, 13):
            buf += "\n".encode("utf-16-le")
        elif ch >= 32:
            buf += data[i:i + 2]
        i += 2
    return buf.decode("utf-16-le", errors="replace")


def hwp_to_text(path):
    ole = olefile.OleFileIO(str(path))
    try:
        header = ole.openstream("FileHeader").read()
        flags = struct.unpack_from("<I", header, 36)[0]
        if flags & 2:
            raise ValueError("암호화된 hwp")
        compressed = bool(flags & 1)
        secs = sorted((e for e in ole.listdir() if e[0] == "BodyText" and e[1].startswith("Section")),
                      key=lambda e: int(e[1][7:]))
        if not secs:
            raise ValueError("BodyText 없음 (배포용 문서일 수 있음)")
        paras = []
        for sec in secs:
            data = ole.openstream(sec).read()
            if compressed:
                data = zlib.decompress(data, -15)
            i = 0
            while i + 4 <= len(data):
                h = struct.unpack_from("<I", data, i)[0]
                tag, size = h & 0x3FF, (h >> 20) & 0xFFF
                i += 4
                if size == 0xFFF:
                    size = struct.unpack_from("<I", data, i)[0]
                    i += 4
                if tag == PARA_TEXT:
                    paras.append(_para_text(data[i:i + size]).strip("\n"))
                i += size
        return "\n".join(paras)
    finally:
        ole.close()


def hwpx_to_text(path):
    paras = []
    with zipfile.ZipFile(path) as z:
        names = sorted((n for n in z.namelist() if re.match(r"Contents/section\d+\.xml$", n)),
                       key=lambda n: int(re.search(r"section(\d+)", n).group(1)))
        for n in names:
            root = ET.fromstring(z.read(n))
            for p in root.iter():
                if p.tag.endswith("}p"):
                    # 표 안 문단은 따로 p로 잡히므로 이 문단의 직계 run 글만 모은다
                    paras.append("".join(t.text or "" for run in p if run.tag.endswith("}run")
                                         for t in run if t.tag.endswith("}t")))
    return "\n".join(paras)


def to_text(path):
    ext = str(path).rsplit(".", 1)[-1].lower()
    if ext == "hwp":
        return hwp_to_text(path)
    if ext == "hwpx":
        return hwpx_to_text(path)
    raise ValueError(f"글 추출 불가 형식: {ext}")
