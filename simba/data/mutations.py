"""Parsers for the mutation-notation strings used by SKEMPI2 and AB-Bind."""

from __future__ import annotations

import re
from dataclasses import dataclass

_SKEMPI_SITE_RE = re.compile(r"^([A-Za-z])([A-Za-z])(\d+)([a-z]?)([A-Za-z])$")
_ABBIND_SITE_RE = re.compile(r"^([A-Za-z])(\d+)([a-z]?)([A-Za-z])$")
_ABBIND_SITE_SCAN_RE = re.compile(r"([A-Za-z]):([A-Z])(\d+)([A-Za-z]?)([A-Z])(?=[A-Za-z]:|,|$)")


@dataclass(frozen=True)
class MutationSite:
    chain: str
    wt_aa: str
    resnum: int
    icode: str  # "" if none
    mut_aa: str


def parse_skempi_mutation_string(s: str) -> list[MutationSite]:
    """Parse e.g. 'RI67A,RI65A' or 'RD100bA' into MutationSite entries."""
    sites = []
    for token in s.split(","):
        token = token.strip()
        m = _SKEMPI_SITE_RE.match(token)
        if not m:
            raise ValueError(f"unparseable SKEMPI mutation token: {token!r}")
        wt_aa, chain, resnum, icode, mut_aa = m.groups()
        sites.append(
            MutationSite(chain=chain, wt_aa=wt_aa.upper(), resnum=int(resnum),
                         icode=icode, mut_aa=mut_aa.upper())
        )
    return sites


def parse_concatenated_abbind_mutation_string(s: str) -> list[MutationSite]:
    """Some redistributions of AB-Bind-style data (e.g. ProtAttBA's AB1101.csv)
    join multi-point mutation tokens with no separator at all, e.g.
    'D:L483TD:V486PD:H487A' instead of 'D:L483T,D:V486P,D:H487A'. A regex scan
    for the fixed-shape 'chain:WTposMUT' pattern segments these unambiguously
    (the trailing uppercase mutant letter can't be confused with the next
    site's leading chain letter because that one is always followed by ':').
    """
    matches = list(_ABBIND_SITE_SCAN_RE.finditer(s.strip()))
    if not matches:
        raise ValueError(f"unparseable concatenated AB-Bind mutation string: {s!r}")
    covered = "".join(m.group(0) for m in matches)
    if covered != s.strip().replace(",", ""):
        raise ValueError(f"mutation string not fully consumed by regex scan: {s!r}")
    return [
        MutationSite(chain=m.group(1), wt_aa=m.group(2).upper(), resnum=int(m.group(3)),
                     icode=m.group(4), mut_aa=m.group(5).upper())
        for m in matches
    ]


def parse_abbind_mutation_string(s: str) -> list[MutationSite]:
    """Parse e.g. 'D:L483T,D:V486P' into MutationSite entries."""
    sites = []
    for token in s.split(","):
        token = token.strip()
        chain, _, rest = token.partition(":")
        if not rest:
            raise ValueError(f"unparseable AB-Bind mutation token: {token!r}")
        m = _ABBIND_SITE_RE.match(rest)
        if not m:
            raise ValueError(f"unparseable AB-Bind mutation token: {token!r}")
        wt_aa, resnum, icode, mut_aa = m.groups()
        sites.append(
            MutationSite(chain=chain, wt_aa=wt_aa.upper(), resnum=int(resnum),
                         icode=icode, mut_aa=mut_aa.upper())
        )
    return sites
