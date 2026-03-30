"""ATC-oriented text normalization helpers.

This module adapts the useful normalization ideas from WhisperATC directly into
the local source tree so the project does not depend on a separate retained
copy of upstream helper files.
"""

from __future__ import annotations

import re

try:
    from whisper.normalizers import EnglishTextNormalizer
except ImportError:
    class EnglishTextNormalizer:  # type: ignore[no-redef]
        """Small fallback so the module stays importable without openai-whisper."""

        def __call__(self, text: str) -> str:
            return " ".join(text.split())


_normalizer = EnglishTextNormalizer()

NATO_ALPHABET_MAPPING = {
    "A": "alpha",
    "B": "bravo",
    "C": "charlie",
    "D": "delta",
    "E": "echo",
    "F": "foxtrot",
    "G": "golf",
    "H": "hotel",
    "I": "india",
    "J": "juliett",
    "K": "kilo",
    "L": "lima",
    "M": "mike",
    "N": "november",
    "O": "oscar",
    "P": "papa",
    "Q": "quebec",
    "R": "romeo",
    "S": "sierra",
    "T": "tango",
    "U": "uniform",
    "V": "victor",
    "W": "whiskey",
    "X": "xray",
    "Y": "yankee",
    "Z": "zulu",
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "00": "hundred",
    "000": "thousand",
    ".": "decimal",
    ",": "comma",
    "-": "dash",
}

NATO_SIMILARITIES = {
    "alfa": "alpha",
    "oskar": "oscar",
    "ekko": "echo",
    "gulf": "golf",
}

TERMINOLOGY_MAPPING = {
    "FL": "flight level",
}

TEXT_SIMILARITIES = {
    "descent": "descend",
}

AIRLINES_IATA_CODES = {
    "BA": "british airways",
    "KL": "klm",
    "LH": "lufthansa",
    "EW": "eurowings",
}

AIRLINES_ICAO_CODES = {
    "BAW": "british airways",
    "DLH": "lufthansa",
    "KLM": "klm",
    "EWG": "eurowings",
}

LOCAL_PROMPT_TERMS = (
    "ABNED",
    "ABSAM",
    "ADIKU",
    "ADOMI",
    "ADUNU",
    "AGASO",
    "AGISI",
    "AGISU",
    "AGOGO",
    "AKOXA",
    "AKZOM",
    "ALFEN",
    "ALINA",
    "AMADA",
    "AMEGA",
    "AMGOD",
    "AMREG",
    "AMRIV",
    "AMSOT",
    "ANDIK",
    "ANETS",
    "ANZUL",
    "APVUV",
    "ARBEP",
    "ARNEM",
    "ARTIP",
    "ARWIN",
    "ASBES",
    "ASGOS",
    "ASNOM",
    "ASTUW",
    "ATRIX",
    "ATWIT",
    "BADEX",
    "BAGOV",
    "BAHSI",
    "BAKLU",
    "BANDU",
    "BASGU",
    "BASNO",
    "BATAK",
    "BAXIM",
    "BEDUM",
    "BEKEM",
    "BEKVU",
    "BELAP",
    "BEMTI",
    "BENUX",
    "BERGI",
    "BERIR",
    "BESBU",
    "BESTI",
    "BETUS",
    "BIBIS",
    "BLUSY",
    "BOBMO",
    "BOGRU",
    "BOGTI",
    "BOVCO",
    "BREDA",
    "BRIAR",
    "BUDIP",
    "BUROG",
    "DANUM",
    "DENAG",
    "DENUT",
    "DERUV",
    "DESUL",
    "DEVIG",
    "DEVUT",
    "DEXOR",
    "DIBIR",
    "DIBRU",
    "DIKAT",
    "DIMOX",
    "DINAK",
    "DISRA",
    "DIVPA",
    "DOBAK",
    "DOFMU",
    "DOTIX",
    "EBAGO",
    "EBUSA",
    "EDFOS",
    "EDOXO",
    "EDUBU",
    "EDUMA",
    "EDUPO",
    "EHOJI",
    "EKDAR",
    "EKNON",
    "EKROS",
    "ELBED",
    "ELPAT",
    "ELSIK",
    "ELSUR",
    "EMMUN",
    "ENKOS",
    "EPOXU",
    "ERMUR",
    "ERSUL",
    "ETEBO",
    "ETPOS",
    "EVELI",
    "FAFLO",
    "FEWEX",
    "FLEVO",
    "GALSO",
    "GEMTI",
    "GETSI",
    "GIKOV",
    "GILIV",
    "GILTI",
    "GIROS",
    "GISEB",
    "GOBNO",
    "GODOS",
    "GOHEM",
    "GOLOR",
    "GOTIG",
    "GREFI",
    "GRONY",
    "GULTO",
    "HAMZA",
    "HECTI",
    "HELEN",
    "HELHO",
    "HOXZA",
    "IBALO",
    "IBNOS",
    "IDAKA",
    "IDGOK",
    "IDRID",
    "IFTAZ",
    "IMVUK",
    "INBAM",
    "INDEV",
    "INDIX",
    "INKET",
    "INLOD",
    "INRIP",
    "INVIT",
    "IPMUR",
    "IPTAS",
    "IPVIS",
    "IRDUK",
    "IVLUT",
    "IVNUD",
    "IXUTA",
    "JOPFI",
    "KAKKO",
    "KAROF",
    "KEGIT",
    "KEKIX",
    "KEROR",
    "KOKIP",
    "KOLAG",
    "KOLAV",
    "KONEP",
    "KONOM",
    "KOPAD",
    "KOPFA",
    "KUBAT",
    "KUDAD",
    "KUSON",
    "KUVOS",
    "LABIL",
    "LAMSO",
    "LANSU",
    "LARAS",
    "LARBO",
    "LASEX",
    "LEGBA",
    "LEKKO",
    "LEKSU",
    "LERGO",
    "LEVKI",
    "LIKDO",
    "LILSI",
    "LOCFU",
    "LONAM",
    "LONLU",
    "LOPIK",
    "LUGUM",
    "LUNIX",
    "LUSOR",
    "LUTET",
    "LUTEX",
    "LUTOM",
    "LUVOR",
    "MAPAD",
    "MASOS",
    "MAVAS",
    "MAXUN",
    "MEBOT",
    "MIMVA",
    "MITSA",
    "MODRU",
    "MOKUM",
    "MOLIX",
    "MOMIC",
    "MONIL",
    "NAKON",
    "NAPRO",
    "NARIX",
    "NARSO",
    "NAVAK",
    "NAVPI",
    "NEKAS",
    "NELFE",
    "NEPTU",
    "NETEX",
    "NETOM",
    "NEWCO",
    "NEXAR",
    "NIDOP",
    "NIGUG",
    "NIHOF",
    "NILMI",
    "NIREX",
    "NIRSI",
    "NIXCO",
    "NOFUD",
    "NOGRO",
    "NOLRU",
    "NOPSU",
    "NORKU",
    "NOVEN",
    "NOWIK",
    "NYKER",
    "OBAGU",
    "OBILO",
    "ODASI",
    "ODVIL",
    "OGBOL",
    "OGINA",
    "OKIDU",
    "OKLOV",
    "OKOKO",
    "OLGAX",
    "OLGER",
    "OLWOF",
    "OMASA",
    "OMFAR",
    "OMORU",
    "ORCAV",
    "OSGOS",
    "OSKUR",
    "OSPAV",
    "OSRON",
    "OSTIR",
    "OTMEC",
    "OTSEL",
    "PAPOX",
    "PELUB",
    "PENIM",
    "PEROR",
    "PESER",
    "PETCA",
    "PETIK",
    "PEVAD",
    "PEVOS",
    "PILEV",
    "PIMIP",
    "PINUS",
    "PIPQU",
    "PODOD",
    "PORWA",
    "PUFLA",
    "PUTTY",
    "RAKIX",
    "RAVLO",
    "REDFA",
    "RELBI",
    "RENDI",
    "RENEQ",
    "RENVU",
    "REWIK",
    "RIKOR",
    "RIMBU",
    "RINIS",
    "RIVER",
    "ROBIS",
    "ROBVI",
    "RODIR",
    "ROFAC",
    "ROLDU",
    "ROMIN",
    "RONSA",
    "ROTEK",
    "ROVEN",
    "ROVOX",
    "RUMER",
    "RUSAL",
    "SASKI",
    "SETWO",
    "SIDNI",
    "SIPLO",
    "SITSU",
    "SOFED",
    "SOGPO",
    "SOKSI",
    "SOMEL",
    "SOMEM",
    "SOMVA",
    "SONEB",
    "SONSA",
    "SOPVI",
    "SORAT",
    "SOTAP",
    "SUBEV",
    "SUGOL",
    "SULUT",
    "SUMAS",
    "SUMUM",
    "SUPUR",
    "SUSET",
    "SUTEB",
    "TACHA",
    "TAFTU",
    "TEBRO",
    "TEMLU",
    "TENLI",
    "TEVKA",
    "TIDVO",
    "TILVU",
    "TINIK",
    "TIREP",
    "TOLKO",
    "TOPPA",
    "TORGA",
    "TORNU",
    "TOTNA",
    "TOTSA",
    "TULIP",
    "TUPAK",
    "TUVOX",
    "TUXAR",
    "ULPAT",
    "ULPEN",
    "ULPOM",
    "ULSED",
    "UNATU",
    "UNEXO",
    "UNKAR",
    "UNORA",
    "UNVAR",
    "UPLOS",
    "UTIRA",
    "UVOXI",
    "VALAM",
    "VALKO",
    "VAPEX",
    "VELED",
    "VELNI",
    "VENAV",
    "VEROR",
    "VEXAR",
    "VICOT",
    "VOLLA",
    "WILEM",
    "WINJA",
    "WISPA",
    "WOODY",
    "XAMAN",
    "XEBOT",
    "XEKRI",
    "XENEV",
    "XIDES",
    "XIPTA",
    "XIPTI",
    "XOMBI",
    "XONLO",
    "YENZO",
    "YOGCE",
    "YOJUP",
    "ZITFA",
    "ZOJIK",
    "Amsterdam",
    "Den Helder",
    "Eelde",
    "Eindhoven",
    "Haastrecht",
    "Lelystad",
    "Maastricht",
    "Pampus",
    "Rekken",
    "Rotterdam",
    "Schiphol",
    "Spykerboor",
    "AMS",
    "HDR",
    "EEL",
    "EHV",
    "FRT",
    "FRO",
    "MAS",
    "PAM",
    "RKN",
    "RTM",
    "SPL",
    "SPY",
    "Zandvoort",
    "zero four",
    "oh four",
    "zero six",
    "oh six",
    "zero nine",
    "oh nine",
    "two two",
    "twenty two",
    "two four",
    "twenty four",
    "two seven",
    "twenty seven",
    "one eight left",
    "eighteen left",
    "one eight right",
    "eighteen right",
    "one eight center",
    "eighteen center",
    "three six left",
    "thirty six left",
    "three six right",
    "thirty six right",
    "three six center",
    "thirty six center",
    "golf one",
    "golf two",
    "golf three",
    "golf four",
    "golf five",
    "sierra one",
    "sierra two",
    "sierra three",
    "sierra four",
    "sierra five",
    "sierra six",
    "sierra seven",
    "november one",
    "november two",
    "november three",
    "november four",
    "november five",
    "november nine",
    "echo one",
    "echo two",
    "echo three",
    "echo four",
    "echo five",
    "whiskey one",
    "whiskey two",
    "whiskey three",
    "whiskey four",
    "whiskey five",
    "whiskey six",
    "whiskey seven",
    "whiskey eight",
    "whiskey nine",
    "whiskey ten",
    "whiskey eleven",
    "whiskey twelve",
    "victor one",
    "victor two",
    "victor three",
    "victor four",
)

COMMON_TEXTS = (
    "line up and wait",
    "line up runway",
    "wind is calm",
    "wake turbulence",
    "Airbus",
    "Boeing",
    "Embraer",
    "RNP",
)


def _replace_bracketed(text: str, left: str, right: str) -> str:
    while left in text and right in text:
        start = text.find(left)
        end = text.rfind(right)
        if start == -1 or end == -1 or end < start:
            break
        text = text[:start] + text[end + 1 :]
    return text


def _remove_non_alnum(text: str) -> str:
    return "".join(character for character in text if character.isalnum() or character == " ")


def _separate_numbers_and_text(text: str) -> str:
    return " ".join(re.split(r"(\d+)", text))


def _transform_word(word: str) -> str:
    if word in NATO_ALPHABET_MAPPING:
        return NATO_ALPHABET_MAPPING[word]
    lowered = word.lower()
    if lowered in NATO_SIMILARITIES:
        return NATO_SIMILARITIES[lowered]
    if word in TERMINOLOGY_MAPPING:
        return TERMINOLOGY_MAPPING[word]
    if lowered in TEXT_SIMILARITIES:
        return TEXT_SIMILARITIES[lowered]
    uppered = word.upper()
    if uppered in AIRLINES_IATA_CODES:
        return AIRLINES_IATA_CODES[uppered]
    if uppered in AIRLINES_ICAO_CODES:
        return AIRLINES_ICAO_CODES[uppered]
    return word


def _aerospace_transform(text: str) -> str:
    return " ".join(_transform_word(word) for word in text.split())


def _remove_spoken_separators(text: str) -> str:
    return " ".join(
        word for word in text.split() if word.lower() not in {"decimal", "comma", "point"}
    )


def _split_numbers_into_digits(text: str) -> str:
    words: list[str] = []
    for word in text.split():
        if word.isnumeric():
            words.extend(list(word))
        else:
            words.append(word)
    return " ".join(words)


def _split_greetings(text: str) -> str:
    return text.replace("goodbye", "good bye")


def _standard_words(text: str) -> str:
    text = text.lower()
    text = text.replace("lineup", "line up")
    text = text.replace("centre", "center")
    text = text.replace("k l m", "klm")
    text = text.replace("niner", "nine")
    text = text.replace("x-ray", "xray")
    return text


def normalize_only(text: str) -> str:
    return _normalizer(text)


def normalize_atc_text(text: str) -> str:
    text = _replace_bracketed(text, "[", "]")
    text = _replace_bracketed(text, "<", ">")
    text = _remove_non_alnum(text)
    text = _separate_numbers_and_text(text)
    text = _aerospace_transform(text)
    text = _remove_spoken_separators(text)
    text = _normalizer(text)
    text = _normalizer(text)
    text = _split_numbers_into_digits(text)
    text = _split_greetings(text)
    text = text.lower()
    return _standard_words(text)
