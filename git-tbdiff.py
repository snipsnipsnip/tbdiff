#!/usr/bin/python3

# git-tbdiff: show the difference between two versions of a topic branch
#
# Copyright (c) 2013, Thomas Rast <trast@inf.ethz.ch>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import hungarian  # https://pypi.python.org/pypi/hungarian
import tempfile
import subprocess
import difflib
import numpy as np
import optparse
from collections import defaultdict
from typing import Iterable, List, Tuple, Dict, Union, Optional, IO, TYPE_CHECKING, cast

if TYPE_CHECKING:
    class _Options:
        creation_fudge = float()
        patches = bool()
        color = bool() if 1 else int()
    options = _Options()

parser = optparse.OptionParser()
parser.add_option('--color', default=True, action='store_true', dest='color')
parser.add_option('--no-color', action='store_false', dest='color')
parser.add_option('--dual-color', action='store_const', dest='color', const=2,
                  help='color both diff and diff-between-diffs')
parser.add_option('--no-patches', action='store_false', dest='patches', default=True,
                  help='short format (no diffs)')
parser.add_option('--creation-weight', action='store',
                  dest='creation_fudge', type=float, default=0.6,
                  help='Fudge factor by which creation is weighted [%default]')


def raw_print(buf: bytes) -> None:
    sys.stdout.buffer.write(buf)
    sys.stdout.buffer.write(b'\n')


def die(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(1)


def strip_uninteresting_patch_parts(lines: Iterable[bytes]) -> List[bytes]:
    out = []
    state = 'head'
    for line in lines:
        if line.startswith(b'diff --git'):
            state = 'diff'
            out.append(b'\n')
            out.append(line)
        elif state == 'head':
            if line.startswith(b'Author: '):
                out.append(line)
                out.append(b'\n')
            elif line.startswith(b'    '):
                out.append(line)
        elif state == 'diff':
            if line.startswith(b'index '):
                pass  # skip
            elif line.startswith(b'@@ '):
                out.append(b'@@\n')
            elif line == b'\n':
                # A completely blank (not ' \n', which is context)
                # line is not valid in a diff.  We skip it silently,
                # because this neatly handles the blank separator line
                # between commits in git-log output.
                pass
            else:
                out.append(line)
            continue
    return out


def read_patches(
        rev_list_args: List[str]) -> Tuple[List[bytes], Dict[bytes, List[bytes]]]:
    series = []  # type: List[bytes]
    diffs = {}  # type: Dict[bytes, List[bytes]]
    p = subprocess.Popen(['git', 'log', '--no-color', '-p', '--no-merges',
                          '--reverse', '--date-order',
                          '--decorate=no', '--no-abbrev-commit']
                         + rev_list_args,
                         stdout=subprocess.PIPE)
    sha1 = None
    data = []

    def handle_commit() -> None:
        if sha1 is not None:
            series.append(sha1)
            diffs[sha1] = strip_uninteresting_patch_parts(data)
            del data[:]
    pipe = p.stdout  # type: IO[bytes]
    for line in pipe:
        if line.startswith(b'commit '):
            handle_commit()
            _, sha1 = line.strip().split()
            continue
        data.append(line)
    handle_commit()
    p.wait()
    return series, diffs


def strip_to_diff_parts_1(lines: Iterable[bytes]) -> Iterable[bytes]:
    in_diff = False
    for line in lines:
        if line.startswith(b'diff --git'):
            in_diff = True
        if not in_diff:
            continue
        if line.startswith(b'@@ '):
            continue
        yield line


def strip_to_diff_parts(
        *args: Iterable[bytes], **kwargs: Iterable[bytes]) -> List[bytes]:
    return list(strip_to_diff_parts_1(*args, **kwargs))


def diffsize(lA: Optional[List[bytes]], lB: Optional[List[bytes]]) -> int:
    if not lA and lB:
        return len(strip_to_diff_parts(lB))
    if not lB and lA:
        return len(strip_to_diff_parts(lA))
    if lB and lA:
        lA = strip_to_diff_parts(lA)
        lB = strip_to_diff_parts(lB)
        diff = difflib.diff_bytes(difflib.unified_diff, lA, lB)
        return len(list(diff))
    else:
        raise ValueError()


def commitinfo(sha1: bytes) -> List[bytes]:
    buf = subprocess.check_output(['git', 'log', '--no-color', '--no-walk',
                                   '--pretty=format:%h %s', sha1.decode('ascii')])  # type: bytes
    return buf.strip().split(b' ', 1)


c_reset = b''
c_commit = b''
c_frag = b''
c_old = b''
c_new = b''
c_inv_old = b''
c_inv_new = b''


def get_color(varname: str, default: str) -> bytes:
    buf = subprocess.check_output(
        ['git', 'config', '--get-color', varname, default])  # type: bytes
    return buf


def invert_ansi_color(color: bytes) -> bytes:
    # \e[7;...m chooses the inverse of \e[...m
    # we try to be nice and also support the reverse transformation
    # (from inverse to normal)
    assert color[:2] == b'\x1b['
    i = color.find(b'7;')
    if i >= 0:
        return color[:i] + color[i + 2:]
    return b'\x1b[7;' + color[2:]


def load_colors() -> None:
    global c_reset, c_commit, c_frag, c_new, c_old, c_inv_old, c_inv_new
    c_reset = get_color('', 'reset')
    c_commit = get_color('color.diff.commit', 'yellow dim')
    c_frag = get_color('color.diff.frag', 'magenta')
    c_old = get_color('color.diff.old', 'red')
    c_new = get_color('color.diff.new', 'green')
    c_inv_old = invert_ansi_color(c_old)
    c_inv_new = invert_ansi_color(c_new)


def commitinfo_maybe(cmt: Optional[bytes]) -> Tuple[bytes, bytes]:
    if cmt:
        sha, subj = commitinfo(cmt)
    else:
        sha = 7 * b'-'
        subj = b''
    return sha, subj


def format_commit_line(left_pair: Optional[Tuple[int, bytes]],
                       right_pair: Optional[Tuple[int, bytes]], has_diff: bool=False) -> None:
    if left_pair:
        i, left = left_pair
    if right_pair:
        j, right = right_pair
    left_sha, left_subj = commitinfo_maybe(left_pair and left)
    right_sha, right_subj = commitinfo_maybe(right_pair and right)
    assert left_pair or right_pair
    if left_pair and not right_pair:
        color = c_old
        status = b'<'
    elif right_pair and not left_pair:
        color = c_new
        status = b'>'
    elif has_diff:
        color = c_commit
        status = b'!'
    else:
        color = c_commit
        status = b'='
    fmt = b'%s'  # color
    args = [color]  # type: List[Union[bytes, int]]
    # left coloring
    if status == b'!':
        fmt += c_reset + c_old
    # left num
    fmt += numfmt if left_pair else numdash
    args += [i + 1] if left_pair else []
    # left hash
    fmt += b": %8s"
    args += [left_sha]
    if status == b'!':
        fmt += c_reset + color
    # middle char
    fmt += b" %s "
    args += [status]
    # right coloring
    if status == b'!':
        fmt += c_reset + c_new
    # right num
    fmt += numfmt if right_pair else numdash
    args += [j + 1] if right_pair else []
    # right hash
    fmt += b": %8s"
    args += [right_sha]
    if status == b'!':
        fmt += c_reset + color
    # subject
    fmt += b" %s"
    args += [right_subj if right_pair else left_subj]
    #
    fmt += b"%s"
    args += [c_reset]
    fmtargs = tuple(args)  # type: Iterable[Union[bytes, int]]
    raw_print(fmt % fmtargs)


def compute_matching_assignment(sA: List[bytes], dA: Dict[bytes, List[bytes]],
                                sB: List[bytes], dB: Dict[bytes, List[bytes]]) -> Tuple[List[int], List[int]]:
    la = len(sA)
    lb = len(sB)
    dist = np.zeros((la + lb, la + lb), dtype=np.uint32)
    for i, u in enumerate(sA):
        for j, v in enumerate(sB):
            dist[i, j] = diffsize(dA[u], dB[v])
    # print dist
    for i, u in enumerate(sA):
        for j in range(lb, lb + la):
            dist[i, j] = int(options.creation_fudge * diffsize(dA[u], None))
    for i in range(la, la + lb):
        for j, v in enumerate(sB):
            dist[i, j] = int(options.creation_fudge * diffsize(None, dB[v]))
    lhs, rhs = hungarian.lap(dist)
    return lhs, rhs


def split_away_same_patches(sA: List[bytes], dA: Dict[bytes, List[bytes]], sB: List[bytes],
                            dB: Dict[bytes, List[bytes]]) -> Tuple[List[Optional[int]], List[Optional[int]]]:
    patchesB = defaultdict(list)  # type: Dict[Iterable[bytes], List[int]]
    for j, v in enumerate(sB):
        patchesB[tuple(dB[v])].append(j)
    eqA = []  # type: List[Optional[int]]
    eqB = [None] * len(sB)  # type: List[Optional[int]]
    for i, u in enumerate(sA):
        patch = tuple(dA[u])
        try:
            j = patchesB[patch].pop(0)
        except IndexError:
            eqA.append(None)
            continue
        eqA.append(j)
        eqB[j] = i
    return eqA, eqB


def make_index_map(eqlist: List[Optional[int]],
                   othereqlist: List[Optional[int]]) -> List[int]:
    imap = []
    mapped = 0
    for _, eq in enumerate(eqlist):
        if eq is None:
            imap.append(mapped)
        mapped += 1
    imap.extend(range(mapped, mapped +
                      sum(1 for x in othereqlist if x is None)))
    return imap


def rebuild_match_list(
        eqlist: List[Optional[int]], matchlist: List[int], imap: List[int]) -> List[int]:
    out = []
    match_it = iter(matchlist)
    for i, eq in enumerate(eqlist):
        if eq is None:
            matched = next(match_it)
            out.append(imap[matched])
        else:
            out.append(eq)
    for i in match_it:
        out.append(imap[i])
    return out


def compute_assignment(sA: List[bytes], dA: Dict[bytes, List[bytes]], sB: List[bytes], dB: Dict[bytes, List[bytes]]
                       ) -> List[Union[Tuple[int, None, None], Tuple[None, int, None], Tuple[int, int, List[bytes]]]]:
    pmap = []  # type: List[Union[Tuple[int, None, None], Tuple[None, int, None], Tuple[int, int, List[bytes]]]]
    la = len(sA)
    lb = len(sB)

    # Attempt to greedily assign an exact match with 0 weight (and
    # give the other choices for this commit a very large weight).
    # This speeds up the case where the patches are the same.
    eqA, eqB = split_away_same_patches(sA, dA, sB, dB)
    lhs1, rhs1 = compute_matching_assignment([u for u, e in zip(sA, eqA) if e is None], dA,
                                             [v for v, e in zip(sB, eqB) if e is None], dB)
    imap = make_index_map(eqA, eqB)
    jmap = make_index_map(eqB, eqA)
    lhs = np.array(rebuild_match_list(eqA, lhs1, jmap))
    rhs = np.array(rebuild_match_list(eqB, rhs1, imap))

    # We assume the user is really more interested in the second
    # argument ("newer" version).  To that end, we print the output in
    # the order of the RHS.  To put the LHS commits that are no longer
    # in the RHS into a good place, we place them once we have seen
    # all of their predecessors in the LHS.
    new_on_lhs = (lhs >= lb)[:la]
    lhs_prior_counter = np.arange(la)

    def process_lhs_orphans() -> None:
        while True:
            assert (lhs_prior_counter >= 0).all()
            w = (lhs_prior_counter == 0) & new_on_lhs
            idx = w.nonzero()[0]
            if len(idx) == 0:
                break
            pmap.append((idx[0], None, None))
            new_on_lhs[idx[0]] = False
            lhs_prior_counter[idx[0] + 1:] -= 1

    for j, (u, i) in enumerate(zip(sB, rhs)):
        # now show an RHS commit
        process_lhs_orphans()
        if i < la:
            idiff = list(difflib.diff_bytes(
                difflib.unified_diff, dA[sA[i]], dB[u]))
            pmap.append((i, j, idiff))
            lhs_prior_counter[i + 1:] -= 1
        else:
            pmap.append((None, j, None))
    process_lhs_orphans()

    return pmap


def print_colored_interdiff(idiff: List[bytes]) -> None:
    for line in idiff:
        line = line.rstrip(b'\n')
        if not line:
            raw_print(b'')
            continue
        # Hunk headers in interdiff are always c_frag, and hunk
        # headers in the original patches are left uncolored (too
        # noisy).
        if line.startswith(b'@@'):
            raw_print(b"    %s%s%s" % (c_frag, line, c_reset))
            continue
        # In traditional single-coloring mode, we color according to
        # the interdiff.
        if options.color == True:
            color = b''
            if line.startswith(b'-'):
                color = c_old
            elif line.startswith(b'+'):
                color = c_new
            raw_print(b''.join([b"    ", color, line, c_reset]))
            continue
        # In dual-color mode, we color the first column (changes
        # between patches) in reverse to highlight the interdiff, and
        # the rest of the line (the actual patches) normally to
        # highlight the diffs themselves.
        lead, line = line[:1], line[1:]
        lead_color = b''
        if lead == b'+':
            lead_color = c_inv_new
        elif lead == b'-':
            lead_color = c_inv_old
        main_color = b''
        if line.startswith(b'+'):
            main_color = c_new
        elif line.startswith(b'-'):
            main_color = c_old
        raw_print(b''.join([b"    ",
                            lead_color, lead, c_reset,
                            main_color, line, c_reset]))


def prettyprint_assignment(sA: List[bytes], dA: Dict[bytes, List[bytes]],
                           sB: List[bytes], dB: Dict[bytes, List[bytes]]) -> None:
    assignment = compute_assignment(sA, dA, sB, dB)
    for i, j, idiff in assignment:
        if j is None and i is not None:
            format_commit_line((i, sA[i]), None)
        elif i is None and j is not None:
            format_commit_line(None, (j, sB[j]))
        elif i is not None and j is not None and idiff is not None and len(idiff) == 0:
            format_commit_line((i, sA[i]), (j, sB[j]), has_diff=False)
        elif i is not None and j is not None and idiff is not None:
            format_commit_line((i, sA[i]), (j, sB[j]), has_diff=True)
            if options.patches:
                # starts with --- and +++ lines
                print_colored_interdiff(idiff[2:])
        else:
            assert False


if __name__ == '__main__':
    options, args = cast(Tuple[_Options, List[str]], parser.parse_args())
    if options.color:
        load_colors()
    if len(args) == 2 and '..' in args[0] and '..' in args[1]:
        rangeA = [args[0]]
        rangeB = [args[1]]
    elif len(args) == 1 and '...' in args[0]:
        A, B = args[0].split("...", 1)
        rangeA = [A, '--not', B]
        rangeB = [B, '--not', A]
    elif len(args) == 3:
        rangeA = [args[1], '--not', args[0]]
        rangeB = [args[2], '--not', args[0]]
    else:
        die("usage: %(command)s A..B C..D\n"
            "   or: %(command)s A...B         # short for:  B..A A..B\n"
            "   or: %(command)s base A B      # short for:  base..A base..B" %
            {'command': sys.argv[0]})
    sA, dA = read_patches(rangeA)
    sB, dB = read_patches(rangeB)
    la = len(sA)
    lb = len(sB)
    numwidth = max(len(str(la)), len(str(lb)))
    numfmt = b"%%%dd" % numwidth
    numdash = numwidth * b'-'
    prettyprint_assignment(sA, dA, sB, dB)
