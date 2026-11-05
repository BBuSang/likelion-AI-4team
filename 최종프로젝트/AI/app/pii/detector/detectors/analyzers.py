"""형태소 분석기 교체용 어댑터.

규칙 기반은 분석기에서 세 가지만 쓴다: 조각(form), 품사(tag), 원문 위치(start, len), 사전에 있는 단어인지(known).
분석기마다 품사 체계와 출력 방식이 달라서 여기서 같은 모양(Tok)으로 맞춘다.
- 품사는 세종 품사(NNG·NNP·NNB·J·E·V…)로 맞춤
- 위치: Kiwi 만 위치를 준다. 나머지는 조각을 원문에서 순서대로 찾아 붙인다.
  '했'→'하+았'처럼 원문에 없는 조각으로 바뀐 것은 위치를 못 찾으므로 버린다
- known: Kiwi 는 사전 밖 단어(OOV)를 알려 준다. Komoran 은 NA(분석 불능), Kkma 는 UN(명사 추정) 태그로 대신한다.
  Okt·Hannanum 은 이 정보가 없어 모두 사전에 있는 단어로 취급 → 규칙이 이름 후보를 덜 찾게 된다(비교 포인트)
"""
from collections import namedtuple
from functools import lru_cache

Tok = namedtuple("Tok", "form tag start len known")

ANALYZERS = ["kiwi", "komoran", "kkma", "hannanum", "okt"]

OKT_TAGS = {"Noun": "NNG", "Josa": "JX", "Eomi": "EF", "PreEomi": "EP", "Verb": "VV", "Adjective": "VA",
            "Determiner": "MM", "Adverb": "MAG", "Conjunction": "MAJ", "Exclamation": "IC", "Suffix": "XSN",
            "Number": "SN", "Alpha": "SL", "Foreign": "SL", "Punctuation": "SF", "KoreanParticle": "SW",
            "Unknown": "NNG"}
HANNANUM_TAGS = {"NC": "NNG", "NQ": "NNP", "NB": "NNB", "NN": "NR", "NP": "NP", "PV": "VV", "PA": "VA",
                 "PX": "VX", "MM": "MM", "MA": "MAG", "II": "IC", "XP": "XPN", "XS": "XSN", "F": "SL"}


def _sejong(name, tag):
    if name == "okt":
        return OKT_TAGS.get(tag, "SW")
    if name == "hannanum":
        if tag[0] in "JE":
            return tag  # JX, JC, JP, EP, EF, EC, ET…
        return HANNANUM_TAGS.get(tag, HANNANUM_TAGS.get(tag[:2], "SW"))
    if name == "kkma":
        if tag == "NNM":
            return "NNB"
        if tag == "UN":
            return "NNG"
        if tag.startswith("MD"):
            return "MM"
    return tag


class Analyzer:
    def __init__(self, name="kiwi"):
        self.name = name
        if name == "kiwi":
            from kiwipiepy import Kiwi
            self._kiwi = Kiwi()
            self._unk = self._kiwi.tokenize("뷁뛟꽑")[0].id
        else:
            import konlpy.tag as kt
            self._tagger = {"komoran": kt.Komoran, "kkma": kt.Kkma,
                            "hannanum": kt.Hannanum, "okt": kt.Okt}[name]()

    @lru_cache(maxsize=100_000)
    def tokenize(self, text: str) -> tuple:
        if self.name == "kiwi":
            return tuple(Tok(t.form, t.tag, t.start, t.len, t.id != self._unk) for t in self._kiwi.tokenize(text))
        # KoNLPy 분석기는 줄 단위로 넣고 위치를 맞춘다 (긴 입력에서 느리거나 멈추는 문제도 줄어든다)
        out, offset = [], 0
        for line in text.split("\n"):
            if line.strip():
                out += self._align(line, offset)
            offset += len(line) + 1
        return tuple(out)

    def _pos(self, line):
        try:
            if self.name == "hannanum":
                return self._tagger.pos(line, ntags=22)
            if self.name == "okt":
                return self._tagger.pos(line, norm=False, stem=False)
            return self._tagger.pos(line)
        except Exception:
            return []  # 일부 분석기는 특수문자 조합에서 예외를 낸다 → 그 줄은 분석 결과 없음

    def _align(self, line, offset):
        toks, cur = [], 0
        for form, tag in self._pos(line):
            idx = line.find(form, cur)
            # 사이에 공백만 있을 때만 같은 조각으로 인정 (원문에 없는 조각으로 바뀐 경우는 버림)
            if idx < 0 or line[cur:idx].strip():
                continue
            known = not ((self.name == "komoran" and tag in ("NA", "NF", "NV")) or (self.name == "kkma" and tag == "UN"))
            toks.append(Tok(form, _sejong(self.name, tag), offset + idx, len(form), known))
            cur = idx + len(form)
        return toks
