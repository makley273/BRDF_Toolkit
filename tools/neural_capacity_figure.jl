#!/usr/bin/env julia
"""
tools/neural_capacity_figure.jl — contact sheet of the neural capacity sweep.

Builds the figure behind "Neural Model Capacity" (`sec:results-capacity` in
`.thesis/latex/content/results.tex`): one row per material, one column per
network architecture, with the measured reference in the leading column, so the
capacity curve of `tab:capacity` can be read visually instead of numerically.

    julia --project=. tools/neural_capacity_figure.jl <bundle-dir> [<bundle-dir> ...] [-o out.png]

Each `<bundle-dir>` is a single-material `tools/fit_material.jl` run that swept
`--neural-hidden` (an "ns sweep"), so it holds one neural row per architecture in
`metrics.csv` and the matching sphere renders in `renders/`. Rows appear in the
order the bundles are given. Architectures are ordered by weight count, which is
the capacity axis the figure is about.

Every panel is annotated with its weight count and its mean image-space SSIM,
the metric chosen over the BRDF-space ones because the figure is itself a set of
renders, so the number and the picture answer the same question. Each row is
labelled with the material and the neural input mode (anisotropic or isotropic),
since that mode decides both the weight count and, on anisotropic materials, the
achievable error floor.
"""

using CairoMakie
using JpegTurbo
using Printf

const LIGHT_TAG = "th045_ph060"   # the light every ns-sweep render was taken under

# ============================================================================
# Bundle loading
# ============================================================================

function read_csv(path)
    lines = readlines(path)
    h = split(lines[1], ",")
    idx = Dict(String(h[i]) => i for i in eachindex(h))
    rows = [split(l, ",") for l in lines[2:end] if !isempty(strip(l))]
    filter!(r -> length(r) == length(h), rows)
    return idx, rows
end

# "21x21" -> 21*21, used only to order the capacity axis.
arch_weight(arch) = prod(parse.(Int, split(arch, 'x')))

"""
    load_bundle(dir) -> NamedTuple

Read one ns-sweep bundle: its material label, neural input mode, and one entry
per architecture holding the render path, the parameter count, and RMS-LogWE.
"""
function load_bundle(dir::AbstractString)
    idx, rows = read_csv(joinpath(dir, "metrics.csv"))
    filter!(r -> r[idx["status"]] == "ok" && r[idx["optimizer"]] == "neural", rows)
    isempty(rows) && error("no successful neural rows in $dir/metrics.csv")

    renders = joinpath(dir, "renders")
    entries = NamedTuple[]
    for r in rows
        arch = String(r[idx["arch"]])
        # The run slug encodes only the factors that varied, so match on the
        # architecture tag rather than reconstructing the whole slug.
        hits = filter(readdir(renders)) do f
            occursin("h-$(arch)__", f) && occursin(LIGHT_TAG, f) && !occursin("__err__", f)
        end
        isempty(hits) && error("no render for arch $arch in $renders")
        push!(entries, (arch = arch,
                        weights = parse(Int, r[idx["n_params"]]),
                        ssim = parse(Float64, r[idx["ssim_mean"]]),
                        path = joinpath(renders, first(hits))))
    end
    sort!(entries; by = e -> e.weights)

    measured = joinpath(renders, "measured__$(LIGHT_TAG).jpg")
    isfile(measured) || error("missing measured render: $measured")

    material = replace(basename(rstrip(dir, ['/', '\\'])), r"^nssweep_" => "")
    aniso = rows[1][idx["aniso"]] == "true"
    return (material = material, aniso = aniso, measured = measured, entries = entries)
end

# ============================================================================
# Figure
# ============================================================================

"""
    build_figure(bundles; panel=190) -> Figure

Grid of sphere renders: rows are materials, columns are the measured reference
followed by each architecture in increasing weight order. `panel` is the side
length in px of one render cell.
"""
function build_figure(bundles::Vector; panel::Int = 190)
    archs = sort(unique(vcat([[e.arch for e in b.entries] for b in bundles]...));
                 by = arch_weight)
    nimg = length(archs) + 1        # image columns: measured + one per architecture
    nrow = length(bundles)

    # Each material occupies two rows, a caption row that sizes itself to the text
    # and an image row pinned to `panel`. Keeping the caption out of the Axis is
    # what makes every render come out at exactly the same size, since an Axis
    # title would otherwise eat into the cell by a different amount per row.
    label_w, pad, gap = 150, 8, 4
    caption_row(i) = 2i
    image_row(i)   = 2i + 1

    # Type is sized relative to the panel, not in absolute points, because the
    # sheet is scaled to \textwidth in the thesis. At seven columns a panel lands
    # near 2 cm wide, so a fixed 11 pt caption would print at about 3 pt.
    head_pt = round(Int, 0.13panel)
    cap_pt  = round(Int, 0.11panel)
    row_pt  = round(Int, 0.10panel)

    fig = Figure(figure_padding = pad)

    # Column headers: the capacity axis.
    Label(fig[1, 2], "Measured"; font = :bold, fontsize = head_pt)
    for (j, a) in enumerate(archs)
        Label(fig[1, j + 2], replace(a, "x" => "×"); font = :bold, fontsize = head_pt)
    end

    for (i, b) in enumerate(bundles)
        mode = b.aniso ? "anisotropic" : "isotropic"
        Label(fig[caption_row(i):image_row(i), 1], "$(b.material)\n($(mode))";
              rotation = pi / 2, fontsize = row_pt, justification = :center)

        draw_panel(fig, caption_row(i), image_row(i), 2, b.measured, "reference", cap_pt)

        by_arch = Dict(e.arch => e for e in b.entries)
        for (j, a) in enumerate(archs)
            e = get(by_arch, a, nothing)
            e === nothing && continue   # architecture not run for this material
            draw_panel(fig, caption_row(i), image_row(i), j + 2, e.path,
                       @sprintf("%d w  ·  %.3f", e.weights, e.ssim), cap_pt)
        end
    end

    colsize!(fig.layout, 1, Fixed(label_w))
    for j in 2:(nimg + 1)
        colsize!(fig.layout, j, Fixed(panel))
    end
    for i in 1:nrow
        rowsize!(fig.layout, image_row(i), Fixed(panel))
    end
    rowgap!(fig.layout, gap)
    colgap!(fig.layout, gap)
    # Caption sits directly on its render, with the breathing room between rows.
    for i in 1:nrow
        rowgap!(fig.layout, caption_row(i), 1)
    end
    resize_to_layout!(fig)
    return fig
end

# One render cell: the caption line, then the image directly under it.
function draw_panel(fig, cap_row::Int, img_row::Int, col::Int,
                    path::AbstractString, caption::AbstractString, cap_pt::Int)
    Label(fig[cap_row, col], caption; fontsize = cap_pt, color = RGBf(0.25, 0.25, 0.25))
    ax = Axis(fig[img_row, col]; aspect = DataAspect())
    image!(ax, rotr90(jpeg_decode(path)))
    hidedecorations!(ax)
    hidespines!(ax)
    return ax
end

# ============================================================================
# CLI
# ============================================================================

function main(argv)
    oi = findfirst(==("-o"), argv)
    out = (oi === nothing || oi == length(argv)) ?
        "exports/fits/neural_capacity.png" : argv[oi + 1]
    dirs = [a for (k, a) in enumerate(argv)
            if !startswith(a, "-") && (oi === nothing || k != oi + 1)]
    isempty(dirs) && error("usage: julia --project=. tools/neural_capacity_figure.jl " *
                           "<bundle-dir> [<bundle-dir> ...] [-o out.png]")

    bundles = [load_bundle(d) for d in dirs]
    for b in bundles
        println("  $(b.material)  [$(b.aniso ? "aniso" : "iso")]  ",
                "$(length(b.entries)) architecture(s)")
    end

    fig = build_figure(bundles)
    mkpath(dirname(out))
    save(out, fig)
    println("Wrote ", out)
    return out
end

main(ARGS)
