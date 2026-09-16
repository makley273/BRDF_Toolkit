#!/usr/bin/env julia
"""
tools/splice_runs.jl — replace one factor level in a sweep bundle with rows
re-fitted after a code change.

    julia --project=. tools/splice_runs.jl <base-bundle> <new-bundle> \\
        --key optimizer --value lbfgsb --out <dir>

Writes `<dir>/metrics_all.csv` holding every row of `<base-bundle>` whose
`<key>` is NOT `<value>`, plus every row of `<new-bundle>` (which supplies the
re-fitted level). The result is a drop-in bundle for `tools/aggregate_sweep.jl`.

This exists because a sweep is expensive and a fix usually touches one backend.
Re-running only the affected level and splicing keeps the untouched levels
bit-identical to what was already reported, so a table can be updated without
silently moving numbers that nothing changed.

The splice is refused if the two bundles do not cover the same materials, since
the paired averaging downstream would then compare levels on different sets.
"""

using Printf

function read_csv(path)
    lines = readlines(path)
    h = String.(split(lines[1], ","))
    rows = [String.(split(l, ",")) for l in lines[2:end] if !isempty(strip(l))]
    filter!(r -> length(r) == length(h), rows)
    return h, rows
end

_arg(flag, default = nothing) = (i = findfirst(==(flag), ARGS);
    i === nothing || i == length(ARGS) ? default : ARGS[i+1])

length(ARGS) < 2 && error("usage: splice_runs.jl <base-bundle> <new-bundle> --key K --value V --out DIR")
base_dir, new_dir = ARGS[1], ARGS[2]
key   = _arg("--key", "optimizer")
value = _arg("--value", nothing)
out   = _arg("--out", nothing)
(value === nothing || out === nothing) && error("--value and --out are required")

hb, rb = read_csv(joinpath(base_dir, "metrics_all.csv"))
hn, rn = read_csv(joinpath(new_dir,  "metrics_all.csv"))
hb == hn || error("column layouts differ between bundles")
ib = Dict(hb[i] => i for i in eachindex(hb))
haskey(ib, key) || error("no column \"$key\"")

mats_b = Set(r[ib["material"]] for r in rb)
mats_n = Set(r[ib["material"]] for r in rn)
if mats_b != mats_n
    only_b = sort(collect(setdiff(mats_b, mats_n)))
    only_n = sort(collect(setdiff(mats_n, mats_b)))
    error("material sets differ. base-only: $(length(only_b)) $(first(only_b, 3))..., " *
          "new-only: $(length(only_n)) $(first(only_n, 3))...")
end

rename = _arg("--rename", nothing)
matched = count(r -> r[ib[key]] == value, rb)
matched == 0 && @warn "no rows matched $key=$value in the base bundle, nothing replaced"

if rename === nothing
    kept = [r for r in rb if r[ib[key]] != value]
else
    # Keep the old rows as their own level. Used when a code change did not fix a
    # backend but forked it: the previous behaviour still exists under a new name,
    # so its measurements stay valid and the two become comparable levels.
    any(r -> r[ib[key]] == rename, rb) &&
        error("level \"$rename\" already exists in the base bundle, refusing to merge two levels")
    kept = [(r[ib[key]] == value ? (c = copy(r); c[ib[key]] = rename; c) : r) for r in rb]
end
dropped = rename === nothing ? matched : 0

mkpath(out)
open(joinpath(out, "metrics_all.csv"), "w") do io
    println(io, join(hb, ","))
    for r in vcat(kept, rn)
        println(io, join(r, ","))
    end
end

if rename === nothing
    @printf("base %d rows, dropped %d with %s=%s, added %d re-fitted -> %d rows over %d materials\n",
            length(rb), dropped, key, value, length(rn), length(kept) + length(rn), length(mats_b))
else
    @printf("base %d rows, relabelled %d from %s=%s to %s, added %d re-fitted -> %d rows over %d materials\n",
            length(rb), matched, key, value, rename, length(rn),
            length(kept) + length(rn), length(mats_b))
end
println("wrote ", joinpath(out, "metrics_all.csv"))
