#!/usr/bin/env julia
"""
RQ5 review app — human judgement pass over the fit bundle.

Reads the bundle written by `tools/rq5_gather.jl` and shows one material at a
time: the measured sphere next to the best analytical and best neural fit, with
their parameters and metrics. You record a verdict, it is written to
`<bundle>/verdicts.csv` immediately, and the app moves on.

Verdicts (RQ5: can the measured data be replaced by a fitted model?)
    analytical  the analytical fit is the better match
    neural      the neural fit is the better match
    equal       both are equally good
    neither     neither is a good enough match to replace the data

Optionally also a quality flag (good / borderline / bad) and a free-text note.

Usage
-----
    julia --project=. tools/rq5_review_app.jl [--bundle exports/rq5_review] [--port 8070]

Keyboard: 1 analytical · 2 neural · 3 equal · 4 neither · ←/→ prev/next · g toggle gain
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

using Bonito
using JSON3
using Base64
using Printf

# ============================================================================
# Bundle loading
# ============================================================================

struct Bundle
    dir::String
    records::Vector{Any}
end

function load_bundle(dir::String)
    path = joinpath(dir, "records.jsonl")
    isfile(path) || error("no records.jsonl in $dir — run tools/rq5_gather.jl first")
    recs = Any[]
    for line in eachline(path)
        isempty(strip(line)) && continue
        try
            push!(recs, JSON3.read(line))
        catch e
            @warn "skipping unreadable record" exception = e
        end
    end
    isempty(recs) && error("records.jsonl in $dir is empty")
    sort!(recs, by = r -> (String(r["dataset"]), String(r["material"])))
    return Bundle(dir, recs)
end

# ============================================================================
# Verdict persistence
# ============================================================================

const VERDICT_HEADER = "key,dataset,material,verdict,quality,note,analytical_model,analytical_ssim,neural_ssim,timestamp"

_csv_escape(s) = '"' * replace(string(s), '"' => "\"\"") * '"'

function load_verdicts(dir::String)
    path = joinpath(dir, "verdicts.csv")
    out = Dict{String, NamedTuple{(:verdict, :quality, :note), Tuple{String, String, String}}}()
    isfile(path) || return out
    lines = readlines(path)
    length(lines) <= 1 && return out
    for line in lines[2:end]
        isempty(strip(line)) && continue
        f = _parse_csv_row(line)
        length(f) < 6 && continue
        out[f[1]] = (verdict = f[4], quality = f[5], note = f[6])
    end
    return out
end

"Minimal RFC4180 row splitter — enough for the fields this app writes."
function _parse_csv_row(line::AbstractString)
    fields = String[]; buf = IOBuffer(); inq = false; i = 1
    chars = collect(line)
    while i <= length(chars)
        c = chars[i]
        if inq
            if c == '"'
                if i < length(chars) && chars[i+1] == '"'
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
            push!(fields, String(take!(buf)))
        else
            write(buf, c)
        end
        i += 1
    end
    push!(fields, String(take!(buf)))
    return fields
end

"Rewrite the whole file — a few hundred rows, so simplicity beats appending."
function save_verdicts(bundle::Bundle, verdicts::Dict)
    path = joinpath(bundle.dir, "verdicts.csv")
    open(path, "w") do io
        println(io, VERDICT_HEADER)
        for rec in bundle.records
            key = String(rec["key"])
            haskey(verdicts, key) || continue
            v = verdicts[key]
            a = rec["analytical"]; n = rec["neural"]
            println(io, join([
                _csv_escape(key),
                _csv_escape(rec["dataset"]),
                _csv_escape(rec["material"]),
                _csv_escape(v.verdict),
                _csv_escape(v.quality),
                _csv_escape(v.note),
                _csv_escape(a === nothing ? "" : a["model"]),
                a === nothing ? "" : @sprintf("%.6f", a["ssim_mean"]),
                n === nothing ? "" : @sprintf("%.6f", n["ssim_mean"]),
                _csv_escape(string(round(Int, time()))),
            ], ","))
        end
    end
    return path
end

# ============================================================================
# Rendering helpers
# ============================================================================

function img_tag(bundle::Bundle, rec, which::String, gain::String; height = 250)
    node = which == "measured" ? rec["measured"] : rec[which]
    node === nothing && return DOM.div("no fit";
        style = "height:$(height)px;display:flex;align-items:center;justify-content:center;" *
                "color:#c0392b;background:#2b2b2b;border-radius:6px;")
    renders = node["renders"]
    haskey(renders, gain) || return DOM.div("no render";
        style = "height:$(height)px;color:#c0392b;")
    path = joinpath(bundle.dir, "renders", String(rec["key"]), String(renders[gain]))
    isfile(path) || return DOM.div("missing file";
        style = "height:$(height)px;color:#c0392b;")
    b64 = base64encode(read(path))
    return DOM.img(src = "data:image/jpeg;base64,$b64",
                   style = "height:$(height)px;width:$(height)px;object-fit:contain;" *
                           "background:#202020;border-radius:6px;display:block;")
end

_fmt(x) = x === nothing || (x isa Real && isnan(x)) ? "--" : @sprintf("%.4f", x)
_fmt6(x) = x === nothing || (x isa Real && isnan(x)) ? "--" : @sprintf("%.6f", x)

function param_lines(node)
    node === nothing && return DOM.div()
    ps = node["params"]
    rows = Any[]
    for k in sort(collect(String.(keys(ps))))
        v = ps[k]
        txt = v isa AbstractVector ? "[" * join([@sprintf("%.4f", x) for x in v], ", ") * "]" :
              v isa Real ? @sprintf("%.4f", v) : string(v)
        push!(rows, DOM.div(
            DOM.span(k; style = "color:#7f8c8d;"),
            DOM.span(" $txt"; style = "color:#ecf0f1;font-family:monospace;");
            style = "font-size:0.78em;line-height:1.5;"))
    end
    return DOM.div(rows...)
end

function metric_block(node, label::String)
    node === nothing && return DOM.div("— fit failed —"; style = "color:#c0392b;")
    return DOM.div(
        DOM.div(DOM.b(String(node["model"])),
                String(node["variant"]) == "--" ? "" : DOM.span(" ($(node["variant"]))";
                    style = "color:#95a5a6;");
                style = "font-size:0.95em;margin-bottom:2px;"),
        DOM.div("$(node["n_params"]) params";
                style = "color:#95a5a6;font-size:0.78em;margin-bottom:6px;"),
        DOM.div(
            DOM.div("SSIM mean  " * _fmt(node["ssim_mean"]);
                    style = "font-family:monospace;font-size:0.82em;color:#2ecc71;"),
            DOM.div("SSIM min   " * _fmt(node["ssim_min"]);
                    style = "font-family:monospace;font-size:0.82em;color:#27ae60;"),
            DOM.div("RMS-LogWE  " * _fmt6(node["rms_logwe"]);
                    style = "font-family:monospace;font-size:0.82em;color:#f39c12;"),
            DOM.div("RMS-CRWE   " * _fmt6(node["rms_crwe"]);
                    style = "font-family:monospace;font-size:0.82em;color:#e67e22;"),
            DOM.div("RMSE       " * _fmt6(node["rmse"]);
                    style = "font-family:monospace;font-size:0.82em;color:#95a5a6;");
            style = "margin-bottom:8px;"),
        param_lines(node),
    )
end

const COL = "flex:1;min-width:260px;display:flex;flex-direction:column;gap:8px;align-items:center;"
const CARD = "background:#34495e;border-radius:8px;padding:12px;"

function column(bundle, rec, which, gain, title, titlecolor)
    return DOM.div(
        DOM.div(title; style = "font-weight:bold;color:$titlecolor;font-size:0.95em;"),
        img_tag(bundle, rec, which, gain),
        DOM.div(which == "measured" ?
                    DOM.div(DOM.div("reference"; style = "color:#95a5a6;font-size:0.85em;"),
                            DOM.div("$(rec["n_samples"]) samples";
                                    style = "color:#7f8c8d;font-size:0.78em;")) :
                    metric_block(rec[which], which);
                style = CARD * "width:100%;min-height:150px;");
        style = COL)
end

# ============================================================================
# App
# ============================================================================

const VERDICT_OPTS = [
    ("analytical", "1 · Analytical better", "#3498db"),
    ("neural",     "2 · Neural better",     "#9b59b6"),
    ("equal",      "3 · Both equally good", "#2ecc71"),
    ("neither",    "4 · Neither good",      "#e74c3c"),
]

const QUALITY_OPTS = [("good", "good"), ("borderline", "borderline"), ("bad", "bad")]

function create_review_app(bundle::Bundle)
    verdicts = load_verdicts(bundle.dir)
    n = length(bundle.records)

    return App() do session
        idx   = Observable(1)
        gain  = Observable("g1")
        note  = Observable("")
        saved = Observable(length(verdicts))
        # bumped whenever a verdict changes, so the card re-renders its highlight
        rev   = Observable(0)

        # Jump to the first material without a verdict.
        let first_todo = findfirst(r -> !haskey(verdicts, String(r["key"])), bundle.records)
            first_todo === nothing || (idx[] = first_todo)
        end

        function record_verdict(v::String)
            rec = bundle.records[idx[]]
            key = String(rec["key"])
            prev = get(verdicts, key, (verdict = "", quality = "", note = ""))
            verdicts[key] = (verdict = v, quality = prev.quality, note = note[])
            save_verdicts(bundle, verdicts)
            saved[] = length(verdicts)
            rev[] = rev[] + 1
            idx[] < n && (idx[] = idx[] + 1)     # auto-advance
        end

        function record_quality(q::String)
            rec = bundle.records[idx[]]
            key = String(rec["key"])
            prev = get(verdicts, key, (verdict = "", quality = "", note = ""))
            verdicts[key] = (verdict = prev.verdict, quality = q, note = note[])
            save_verdicts(bundle, verdicts)
            saved[] = length(verdicts)
            rev[] = rev[] + 1
        end

        # --- controls -------------------------------------------------------
        verdict_trigger = Observable("")
        on(verdict_trigger) do v
            isempty(v) || record_verdict(v)
        end
        quality_trigger = Observable("")
        on(quality_trigger) do q
            isempty(q) || record_quality(q)
        end
        nav_trigger = Observable(0)
        on(nav_trigger) do d
            d == 0 && return
            idx[] = clamp(idx[] + d, 1, n)
        end
        gain_trigger = Observable(0)
        on(gain_trigger) do _
            gain[] = gain[] == "g1" ? "g5" : "g1"
        end

        # Clear the note box whenever we move to another material.
        on(idx) do _
            note[] = ""
        end

        btn(label, color, trigger, value) = DOM.button(
            label;
            onclick = js"""function(e){ $(trigger).notify($(value)); }""",
            style = "background:$color;color:white;border:none;border-radius:6px;" *
                    "padding:10px 16px;font-size:0.9em;font-weight:bold;cursor:pointer;")

        verdict_row = DOM.div(
            [btn(lbl, col, verdict_trigger, v) for (v, lbl, col) in VERDICT_OPTS]...;
            style = "display:flex;gap:10px;flex-wrap:wrap;")

        quality_row = DOM.div(
            DOM.span("quality: "; style = "color:#95a5a6;font-size:0.85em;"),
            [btn(lbl, "#7f8c8d", quality_trigger, q) for (q, lbl) in QUALITY_OPTS]...;
            style = "display:flex;gap:8px;align-items:center;")

        note_box = DOM.input(
            type = "text", placeholder = "note (optional) — saved with the next verdict click",
            oninput = js"""function(e){ $(note).notify(e.srcElement.value); }""",
            style = "flex:1;min-width:320px;padding:8px 10px;border-radius:6px;" *
                    "border:1px solid #7f8c8d;background:#2c3e50;color:#ecf0f1;font-size:0.85em;")

        nav_row = DOM.div(
            btn("← prev", "#7f8c8d", nav_trigger, -1),
            btn("next →", "#7f8c8d", nav_trigger, 1),
            btn("toggle gain", "#16a085", gain_trigger, 1),
            note_box;
            style = "display:flex;gap:10px;align-items:center;flex-wrap:wrap;")

        # --- reactive card --------------------------------------------------
        card = map(idx, gain, rev) do i, g, _
            rec = bundle.records[i]
            key = String(rec["key"])
            cur = get(verdicts, key, nothing)
            badge = cur === nothing ?
                DOM.span("unjudged"; style = "color:#95a5a6;font-size:0.85em;") :
                DOM.span("verdict: $(cur.verdict)" *
                         (isempty(cur.quality) ? "" : " · $(cur.quality)") *
                         (isempty(cur.note) ? "" : " · “$(cur.note)”");
                         style = "color:#2ecc71;font-size:0.85em;font-weight:bold;")

            DOM.div(
                DOM.div(
                    DOM.div(
                        DOM.span("$(rec["material"])";
                                 style = "font-size:1.25em;font-weight:bold;color:#ecf0f1;"),
                        DOM.span("  [$(rec["dataset"])]";
                                 style = "color:#95a5a6;font-size:0.9em;")),
                    DOM.div(badge);
                    style = "display:flex;justify-content:space-between;align-items:center;" *
                            "margin-bottom:4px;"),
                DOM.div("gain $(g == "g1" ? "×1" : "×5")  ·  light θ=45° φ=60°  ·  " *
                        "SSIM averaged over all 3 lights";
                        style = "color:#7f8c8d;font-size:0.78em;margin-bottom:12px;"),
                DOM.div(
                    column(bundle, rec, "measured",   g, "Measured",   "#ecf0f1"),
                    column(bundle, rec, "analytical", g, "Analytical", "#3498db"),
                    column(bundle, rec, "neural",     g, "Neural",     "#9b59b6");
                    style = "display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start;"),
            )
        end

        progress = map(idx, saved) do i, s
            DOM.div("material $i / $n   ·   $s judged   ·   $(n - s) left";
                    style = "color:#95a5a6;font-size:0.85em;")
        end

        # Keyboard shortcuts.
        keys = DOM.script(js"""
            document.addEventListener('keydown', function(e){
                if (e.target.tagName === 'INPUT') return;
                if (e.key === '1') $(verdict_trigger).notify('analytical');
                else if (e.key === '2') $(verdict_trigger).notify('neural');
                else if (e.key === '3') $(verdict_trigger).notify('equal');
                else if (e.key === '4') $(verdict_trigger).notify('neither');
                else if (e.key === 'ArrowLeft')  $(nav_trigger).notify(-1);
                else if (e.key === 'ArrowRight') $(nav_trigger).notify(1);
                else if (e.key === 'g') $(gain_trigger).notify(Date.now());
                else return;
                e.preventDefault();
            });
        """)

        return DOM.div(
            DOM.div(
                DOM.h2("RQ5 fit review";
                       style = "margin:0 0 2px 0;color:#ecf0f1;font-size:1.4em;"),
                progress;
                style = "margin-bottom:14px;"),
            card,
            DOM.div(verdict_row, quality_row, nav_row;
                    style = "display:flex;flex-direction:column;gap:10px;margin-top:16px;" *
                            "padding-top:14px;border-top:1px solid #4a6278;"),
            DOM.div("verdicts → $(joinpath(bundle.dir, "verdicts.csv"))";
                    style = "color:#7f8c8d;font-size:0.75em;margin-top:12px;"),
            keys;
            style = "font-family:system-ui,-apple-system,sans-serif;background:#2c3e50;" *
                    "padding:20px;min-height:100vh;box-sizing:border-box;")
    end
end

# ============================================================================
# Entry point
# ============================================================================

function main(argv)
    opts = Dict("bundle" => joinpath("exports", "rq5_review"),
                "port" => "8070", "host" => "0.0.0.0")
    i = 1
    while i < length(argv)
        startswith(argv[i], "--") || error("unrecognised argument: $(argv[i])")
        opts[argv[i][3:end]] = argv[i+1]; i += 2
    end

    bundle = load_bundle(opts["bundle"])
    port   = parse(Int, opts["port"])
    host   = opts["host"]

    done = length(load_verdicts(bundle.dir))
    println("=" ^ 60)
    println("RQ5 fit review")
    println("=" ^ 60)
    println("bundle:   $(bundle.dir)")
    println("materials: $(length(bundle.records))  ($done already judged)")
    println("local:    http://localhost:$port")
    println("keys:     1 analytical · 2 neural · 3 equal · 4 neither · ←/→ · g gain")
    println("Ctrl+C to stop")
    println("=" ^ 60)

    server = Bonito.Server(create_review_app(bundle), host, port; proxy_url = ".")
    wait(server)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
