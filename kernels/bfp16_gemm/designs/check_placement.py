"""check_placement.py <final.prj>: whether a whole_array_bfp_colwise build is
laid out as its runtime sequence (PLACEMENT in the design) and the host's
instruction streams (src/npu/bfp16_insts.cpp) take it to be. The toolchain
chooses shim channels, memtile channels, descriptor and lock IDs and buffer
addresses; they have come out the same for every shape and option, and this
fails the build the day they don't.
"""
import re
import sys

import whole_array_bfp_colwise as d

RTP = 0xFC00  # the cores' runtime parameters (src/npu/insts_stream.h)


def main():
    src = open(sys.argv[1] + "/input_with_addresses.mlir").read()
    bad = []

    def want(what, got, expected):
        if got != expected:
            bad.append(f"{what} is {got}, not {expected}")

    # shim channels: column c's A on its own shim's MM2S 0 and its C on that
    # shim's S2MM 0; core row r's B on shim 2r's MM2S 1
    shim = {
        m.group(1): (int(m.group(2)), m.group(3), int(m.group(4)))
        for m in re.finditer(r"@(\w+?)_shim_alloc\(%shim_noc_tile_(\d)_0, (\w+), (\d)\)", src)
    }
    for c in range(d.COLS):
        want(f"A_L3L2_{c}'s shim channel", shim.get(f"A_L3L2_{c}"), (c, "MM2S", 0))
        want(f"C_L2L3_{c}'s shim channel", shim.get(f"C_L2L3_{c}"), (c, "S2MM", 0))
    for r in range(d.ROWS):
        want(f"B_L3L1_{r}'s shim channel", shim.get(f"B_L3L1_{r}"), (2 * r, "MM2S", 1))

    locks = {
        m.group(1): (int(m.group(2)), int(m.group(3)), int(m.group(4)))
        for m in re.finditer(r"%(\w+) = aie\.lock\(%mem_tile_(\d)_1, (\d+)\) \{init = (\d+)", src)
    }
    buffers = {
        m.group(3): (int(m.group(1)), int(m.group(2)))
        for m in re.finditer(r'aie\.buffer\(%mem_tile_(\d)_1\) \{address = (\d+)[^\n]*sym_name = "(\w+)"', src)
    }
    for c in range(d.COLS):
        m = re.search(r"aie\.memtile_dma\(%%mem_tile_%d_1\) \{(.*?)\n    \}" % c, src, re.S)
        if not m:
            bad.append(f"memtile {c} has no DMA program")
            continue
        # each descriptor: (channel, fifo, buffer, offset in tiles, id, next id, lock acquired, lock released)
        chan, acq, bds = None, None, []
        for line in m.group(1).splitlines():
            s = re.search(r"aie\.dma_start\((\w+), (\d+)", line)
            if s:
                chan = (s.group(1), int(s.group(2)))
            u = re.search(r"aie\.use_lock\(%(\w+), (\w+),", line)
            if u and u.group(2).startswith("Acquire"):
                acq = locks[u.group(1)][1]
            elif u:
                bds[-1][-1] = locks[u.group(1)][1]
            b = re.search(
                r"aie\.dma_bd\(%(\w+?)_buff_(\d) .*offset = (\d+) len = (\d+)\) \{bd_id = (\d+) : i32, next_bd_id = (\d+)",
                line,
            )
            if b:
                bds.append([chan, b.group(1), int(b.group(2)), int(b.group(3)) // int(b.group(4)), int(b.group(5)),
                            int(b.group(6)), acq, None])
        a_in = [x for x in bds if x[1] == f"A_L3L2_{c}_cons" and x[0][0] == "S2MM"]
        a_out = [x for x in bds if x[1] == f"A_L3L2_{c}_cons" and x[0][0] == "MM2S"]
        c_out = [x for x in bds if x[1] == f"C_L2L3_{c}" and x[0][0] == "MM2S"]
        want(f"memtile {c}: A in (channel, descriptors, locks)",
             [(x[0], x[4], x[5], x[6], x[7]) for x in a_in],
             [(("S2MM", d.A_IN_CH), d.A_IN_BD + i, d.A_IN_BD + 1 - i, d.A_FREE_LOCK, d.A_FULL_LOCK) for i in range(2)])
        want(f"memtile {c}: A out (channel, descriptors, locks)",
             [(x[0], x[4], x[5], x[6], x[7]) for x in a_out],
             [(("MM2S", d.A_OUT_CH), d.A_OUT_BD + i, d.A_OUT_BD + 1 - i, d.A_FULL_LOCK, d.A_FREE_LOCK) for i in range(2)])
        want(f"memtile {c}: C out (channel, core row, descriptor)",
             [(x[0], x[3], x[4]) for x in c_out],
             [(("MM2S", d.C_OUT_CH), i, d.C_OUT_BD + i) for i in range(d.ROWS)])
        want(f"memtile {c}: A's locks' starting values",
             sorted((n, v) for t, n, v in locks.values() if t == c and n in (d.A_FREE_LOCK, d.A_FULL_LOCK)),
             [(d.A_FREE_LOCK, 2), (d.A_FULL_LOCK, 0)])
        # what resident A takes must be free: its descriptors, its locks, and
        # the memory past the C join (A's own tiles may sit there)
        used_bds = {x[4] for x in bds}
        used_locks = {n for t, n, v in locks.values() if t == c}
        want(f"memtile {c}: resident A's descriptors in use by a fifo", used_bds & {d.RES_FILL, d.RES_PLAY, d.RES_LAST}, set())
        want(f"memtile {c}: resident A's locks in use by a fifo", used_locks & {d.RES_FREE_LOCK, d.RES_FULL_LOCK}, set())
        want(f"memtile {c}: buffers past the C join that are not A's tiles",
             sorted(n for n, (t, addr) in buffers.items() if t == c and addr >= d.RES_BASE and not n.startswith(f"A_L3L2_{c}_")), [])
        want(f"memtile {c}: the C join's address", buffers.get(f"C_L2L3_{c}_buff_0"), (c, 0))
    rtp = {int(a) for a in re.findall(r'aie\.buffer\(%tile_\d_\d\) \{address = (\d+)[^\n]*sym_name = "nmid_', src)}
    want("the cores' runtime parameters' address", rtp, {RTP})
    for b in bad:
        print("PLACEMENT:", b, file=sys.stderr)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
