#!/usr/bin/env julia
"""
RQ5 blind A/B app — which analytical fit actually looks closer?

Restricted to the analytical materials where the run with the best mean SSIM is
not the run with the best RMS-LogWE. Both fits are shown either side of the
measured reference and you pick the one that matches it better. Which side
carries which metric's pick is hidden, fixed per material by a hash of its key
so it stays the same across restarts, and never shown in the interface.

Verdicts are written to `<bundle>/ab_verdicts.csv` on every click, together with
the resolved answer, so the outcome can be analysed later without re-judging.

Usage
-----
    julia --project=. tools/rq5_ab_app.jl [--bundle exports/rq5_metrics] [--port 8413]

Keyboard: 1 left  2 right  3 no difference  <-/-> prev/next  g toggle gain
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

using Bonito
using JSON3
using Base64
using Printf

# ============================================================================
# Bundle
# ============================================================================

struct Pair_
    key::String
    dataset::String
    material::String
    left_slug::String        # what is actually shown on the left
    right_slug::String
    left_metric::String      # "ssim" or "logwe", never shown in the UI
    right_metric::String
    renders::Dict{String, Any}
    rank::Dict{String, Any}  # slug -> candidate record
end

struct Bundle
    dir::String
    pairs::Vector{Pair_}
end

"FNV-1a over the key, so the side assignment is stable across sessions."
function stable_hash(s::AbstractString)
    h = 0x811c9dc5
    for b in codeunits(s)
        h = (h ⊻ b) * 0x01000193
        h &= 0xffffffff
    end
    return h
end

function load_bundle(dir::String)
    path = joinpath(dir, "records.jsonl")
    isfile(path) || error("no records.jsonl in $dir — run tools/rq5_gather.jl first")

    pairs = Pair_[]
    for line in eachline(path)
        isempty(strip(line)) && continue
        rec = JSON3.read(line)
        w = rec["winners"]["analytical"]
        (haskey(w, "ssim") && haskey(w, "logwe")) || continue
        s_slug, l_slug = String(w["ssim"]), String(w["logwe"])
        s_slug == l_slug && continue                      # only the disagreements

        rank = Dict{String, Any}()
        for c in rec["ranking"]
            c["family"] == "analytical" && (rank[String(c["slug"])] = c)
        end

        renders = Dict{String, Any}(String(k) => v for (k, v) in rec["candidates"])
        renders["__measured__"] = rec["measured"]["renders"]

        key = String(rec["key"])
        ssim_left = iseven(stable_hash(key))
        push!(pairs, Pair_(key, String(rec["dataset"]), String(rec["material"]),
                           ssim_left ? s_slug : l_slug,
                           ssim_left ? l_slug : s_slug,
                           ssim_left ? "ssim" : "logwe",
                           ssim_left ? "logwe" : "ssim",
                           renders, rank))
    end
    isempty(pairs) && error("no disagreeing materials found in $dir")
    sort!(pairs, by = p -> (p.dataset, p.material))
    return Bundle(dir, pairs)
end

# ============================================================================
# Verdict persistence
# ============================================================================

const HEADER = "key,dataset,material,choice,chosen_metric,left_metric," *
               "ssim_model,logwe_model,ssim_pick_ssim,logwe_pick_ssim," *
               "ssim_pick_logwe,logwe_pick_logwe,timestamp"

_esc(s) = '"' * replace(string(s), '"' => "\"\"") * '"'

function _split_csv(line::AbstractString)
    out = String[]; buf = IOBuffer(); inq = false; i = 1
    ch = collect(line)
    while i <= length(ch)
        c = ch[i]
        if inq
            if c == '"'
                if i < length(ch) && ch[i+1] == '"'
                    write(buf, '"'); i += 1
                else
                    inq = false
                end
            else
                write(buf, c)
            end
        elseif c == '"'
            inq = true
        elseif c == ','
            push!(out, String(take!(buf)))
        else
            write(buf, c)
        end
        i += 1
    end
    push!(out, String(take!(buf)))
    return out
end

function load_verdicts(dir::String)
    path = joinpath(dir, "ab_verdicts.csv")
    out = Dict{String, String}()
    isfile(path) || return out
    lines = readlines(path)
    length(lines) <= 1 && return out
    for l in lines[2:end]
        isempty(strip(l)) && continue
        f = _split_csv(l)
        length(f) >= 4 && (out[f[1]] = f[4])
    end
    return out
end

function save_verdicts(b::Bundle, verdicts::Dict{String, String})
    path = joinpath(b.dir, "ab_verdicts.csv")
    open(path, "w") do io
        println(io, HEADER)
        for p in b.pairs
            haskey(verdicts, p.key) || continue
            choice = verdicts[p.key]
            chosen = choice == "same" ? "same" :
                     choice == "left" ? p.left_metric : p.right_metric
            s_slug = p.left_metric == "ssim" ? p.left_slug : p.right_slug
            l_slug = p.left_metric == "ssim" ? p.right_slug : p.left_slug
            sc, lc = p.rank[s_slug], p.rank[l_slug]
            println(io, join([
                _esc(p.key), _esc(p.dataset), _esc(p.material),
                _esc(choice), _esc(chosen), _esc(p.left_metric),
                _esc(sc["model"]), _esc(lc["model"]),
                @sprintf("%.6f", sc["ssim_mean"]), @sprintf("%.6f", lc["ssim_mean"]),
                @sprintf("%.8f", sc["rms_logwe"]), @sprintf("%.8f", lc["rms_logwe"]),
                _esc(string(round(Int, time()))),
            ], ","))
        end
    end
end

# ============================================================================
# UI
# ============================================================================

function img(b::Bundle, p::Pair_, slug::String, gain::String; h = 300)
    node = get(p.renders, slug, nothing)
    node === nothing && return DOM.div("no render"; style = "color:#c0392b;height:$(h)px;")
    file = get(node, gain, nothing)
    file === nothing && return DOM.div("no gain"; style = "color:#c0392b;height:$(h)px;")
    path = joinpath(b.dir, "renders", p.key, String(file))
    isfile(path) || return DOM.div("missing"; style = "color:#c0392b;height:$(h)px;")
    return DOM.img(src = "data:image/jpeg;base64,$(base64encode(read(path)))",
                   style = "height:$(h)px;width:$(h)px;object-fit:contain;" *
                           "background:#202020;border-radius:6px;display:block;")
end

btn(label, color, trigger, value) = DOM.button(
    label;
    onclick = js"""function(e){ $(trigger).notify($(value)); }""",
    style = "background:$color;color:white;border:none;border-radius:6px;" *
            "padding:12px 20px;font-size:0.95em;font-weight:bold;cursor:pointer;")

function create_ab_app(b::Bundle)
    verdicts = load_verdicts(b.dir)
    n = length(b.pairs)

    return App() do session
        idx  = Observable(1)
        gain = Observable("g1")
        rev  = Observable(0)
        done = Observable(length(verdicts))

        let todo = findfirst(p -> !haskey(verdicts, p.key), b.pairs)
            todo === nothing || (idx[] = todo)
        end

        choice_trigger = Observable("")
        on(choice_trigger) do c
            isempty(c) && return
            verdicts[b.pairs[idx[]].key] = c
            save_verdicts(b, verdicts)
            done[] = length(verdicts)
            rev[] = rev[] + 1
            idx[] < n && (idx[] = idx[] + 1)
        end
        nav = Observable(0)
        on(nav) do d
            d == 0 || (idx[] = clamp(idx[] + d, 1, n))
        end
        gain_t = Observable(0)
        on(gain_t) do _
            gain[] = gain[] == "g1" ? "g5" : "g1"
        end

        card = map(idx, gain, rev) do i, g, _
            p = b.pairs[i]
            cur = get(verdicts, p.key, nothing)
            badge = cur === nothing ?
                DOM.span("unjudged"; style = "color:#95a5a6;font-size:0.85em;") :
                DOM.span("chose: $cur"; style = "color:#2ecc71;font-weight:bold;font-size:0.85em;")

            col(title, node, color) = DOM.div(
                DOM.div(title; style = "font-weight:bold;color:$color;margin-bottom:6px;"),
                node;
                style = "display:flex;flex-direction:column;align-items:center;")

            DOM.div(
                DOM.div(
                    DOM.div(DOM.span(p.material;
                                     style = "font-size:1.25em;font-weight:bold;color:#ecf0f1;"),
                            DOM.span("  [$(p.dataset)]"; style = "color:#95a5a6;")),
                    badge;
                    style = "display:flex;justify-content:space-between;align-items:center;"),
                DOM.div("which fit matches the measured sphere in the middle?  " *
                        "gain $(g == "g1" ? "×1" : "×5")";
                        style = "color:#7f8c8d;font-size:0.8em;margin:4px 0 14px;"),
                DOM.div(
                    col("Left", img(b, p, p.left_slug, g), "#e67e22"),
                    col("Measured", img(b, p, "__measured__", g), "#ecf0f1"),
                    col("Right", img(b, p, p.right_slug, g), "#16a085");
                    style = "display:flex;gap:22px;justify-content:center;flex-wrap:wrap;"),
            )
        end

        progress = map(idx, done) do i, d
            DOM.div("pair $i / $n   ·   $d judged   ·   $(n - d) left";
                    style = "color:#95a5a6;font-size:0.85em;")
        end

        keys = DOM.script(js"""
            document.addEventListener('keydown', function(e){
                if (e.target.tagName === 'INPUT') return;
                if (e.key === '1') $(choice_trigger).notify('left');
                else if (e.key === '2') $(choice_trigger).notify('right');
                else if (e.key === '3') $(choice_trigger).notify('same');
                else if (e.key === 'ArrowLeft')  $(nav).notify(-1);
                else if (e.key === 'ArrowRight') $(nav).notify(1);
                else if (e.key === 'g') $(gain_t).notify(Date.now());
                else return;
                e.preventDefault();
            });
        """)

        return DOM.div(
            DOM.div(DOM.h2("RQ5 blind A/B — analytical fits";
                           style = "margin:0 0 2px;color:#ecf0f1;font-size:1.35em;"),
                    progress; style = "margin-bottom:14px;"),
            card,
            DOM.div(
                btn("1 · Left", "#e67e22", choice_trigger, "left"),
                btn("2 · Right", "#16a085", choice_trigger, "right"),
                btn("3 · No difference", "#7f8c8d", choice_trigger, "same"),
                btn("← prev", "#566573", nav, -1),
                btn("next →", "#566573", nav, 1),
                btn("toggle gain", "#2980b9", gain_t, 1);
                style = "display:flex;gap:10px;flex-wrap:wrap;margin-top:18px;" *
                        "padding-top:14px;border-top:1px solid #4a6278;"),
            DOM.div("verdicts → $(joinpath(b.dir, "ab_verdicts.csv"))";
                    style = "color:#7f8c8d;font-size:0.75em;margin-top:12px;"),
            keys;
            style = "font-family:system-ui,-apple-system,sans-serif;background:#2c3e50;" *
                    "padding:20px;min-height:100vh;box-sizing:border-box;")
    end
end

function main(argv)
    opts = Dict("bundle" => joinpath("exports", "rq5_metrics"),
                "port" => "8413", "host" => "0.0.0.0")
    i = 1
    while i < length(argv)
        startswith(argv[i], "--") || error("unrecognised argument: $(argv[i])")
        opts[argv[i][3:end]] = argv[i+1]; i += 2
    end

    b = load_bundle(opts["bundle"])
    port = parse(Int, opts["port"])
    println("=" ^ 60)
    println("RQ5 blind A/B — analytical fits where SSIM and RMS-LogWE disagree")
    println("=" ^ 60)
    println("bundle: $(b.dir)")
    println("pairs:  $(length(b.pairs))  ($(length(load_verdicts(b.dir))) already judged)")
    println("local:  http://localhost:$port")
    println("keys:   1 left · 2 right · 3 no difference · ←/→ · g gain")
    println("=" ^ 60)

    wait(Bonito.Server(create_ab_app(b), opts["host"], port; proxy_url = "."))
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
