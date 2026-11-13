"""정보 없는 줄 걸러내기: 줄 '전체'가 정해진 문장 틀과 일치할 때만 지운다.
(단어만으로 찾으면 '다만, …한 경우에는 달리 판단될 수 있음'처럼 조건을 담은 판단 문장까지 지워진다)"""
import re

BULLET = re.compile(r"^[\s□■○●ㅇ◎※\-\*·]+")

RULES = {
    # 다른 칸을 가리키기만 하는 줄
    "가리키기": re.compile(
        r"(?:하단(?:의)?|아래(?:의)?)\s*[‘'\"“]?\s*이\s*유\s*[’'\"”]?\s*(?:를\s*)?참조(?:하시기\s*바랍니다)?"
        r"|아래\s*(?:의\s*)?내용을?\s*(?:참고|확인|참조)\s*하시기\s*바랍니다"
        r"|(?:상기\s*)?회답\s*(?:내용\s*)?(?:참조|참고|과\s*동일|과\s*같음|과\s*같습니다)"
        r"|상\s*동|위와\s*같음|위와\s*동일"
    ),
    # 모든 회신에 붙는 정형 면책 문구 (조건·사유를 담은 문장은 틀에 맞지 않아 남음)
    "면책": re.compile(
        r"(?:다만,?\s*)?(?:상기|위)\s*의견은\s*질의(?:하신\s*사항\s*만?을|내용으로\s*파악할\s*수\s*있는\s*사실관계만을)\s*"
        r"토대로\s*(?:실무\s*)?검토한\s*내용이며,?\s*구체적인\s*사실관계(?:\s*등)?(?:\s*제반\s*사정)?에\s*따라(?:서)?\s*(?:이와\s*)?"
        r"다르게\s*해석될\s*(?:여지[가도]?\s*있|수(?:도)?\s*있)[^\n]*"
        r"|(?:다만,?\s*)?질의(?:하신\s*내용은|내용이\s*개별적\s*사실인정에\s*관한\s*사항으로)\s*(?:지급결제\s*동향,\s*)?"
        r"구체적인\s*(?:사실관계\s*등\s*)?제반\s*사정\s*등?에\s*따라\s*달리\s*판단될\s*수\s*있다는\s*점을\s*양지하여\s*주시기\s*바랍니다\.?"
        r"|(?:한편|다만),?\s*이에\s*대해서는\s*구체적인\s*영업방식\s*및\s*행위\s*관련\s*사실관계에\s*따라[^\n]*양지하여\s*주시기\s*바랍니다\.?"
    ),
    # 서명·날짜만 있는 줄
    "서명날짜": re.compile(
        r"금융(?:감독원|위원회)\s*(?:원장|위원장)?\s*\(\s*인\s*\)"
        r"|\d{4}\s*[.년]\s*\d{1,2}\s*[.월]\s*\d{0,2}\s*일?\s*\.?"
    ),
    "인사말": re.compile(r"감사합니다\.?(?:\s*끝\.?)?|끝\.?"),
    # 빈칸 대신 쓴 말 (단독 '없음'·'-'는 실제 답일 수 있어 제외)
    "자리채움": re.compile(r"(?:\d\)\s*)?해당\s*(?:사항\s*)?없음"),
}
# 모든 틀은 끝 마침표를 허용
RULES = {k: re.compile(r"(?:" + p.pattern + r")\s*\.?") for k, p in RULES.items()}
# 문장 끝에 붙은 맺음말만 떼기 ("… 알려드립니다. 감사합니다. 끝.")
TAIL_GREETING = re.compile(r"\s*(?:감사합니다\.?\s*)?끝\.\s*$|\s+감사합니다\.\s*$")


def strip_noise(text, stats=None):
    out = []
    for line in text.split("\n"):
        core = BULLET.sub("", line).strip()
        hit = next((name for name, p in RULES.items() if core and p.fullmatch(core)), None)
        if hit:
            if stats is not None:
                stats[hit] = stats.get(hit, 0) + 1
            continue
        new = TAIL_GREETING.sub("", line)
        if new != line and stats is not None:
            stats["맺음말"] = stats.get("맺음말", 0) + 1
        out.append(new)
    return "\n".join(out)
