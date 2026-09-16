"""
Unified multi-tool server + top navigation bar.

The platform is a set of independent tools (BRDF analysis, Rusinkiewicz, sampling),
each its own `create_*_app`. This file ties them together WITHOUT touching their
internals:

- `_nav_bar`     — a slim top bar prepended to every page so you can switch tools.
- `run_platform` — one `Bonito.Server`. `/` is the interactive BRDF analysis app
                   (the default), and each other tool gets its own route. Each app
                   is wrapped at the route level with the nav bar, so the tool
                   `create_*` functions stay untouched and the single-tool
                   `run_*_app` entry points keep working.

Everything is served from one origin/port, so plain `href="/tool"` links work and
the LAN `proxy_url` (see resolve_proxy_url) is resolved once.

Location: src/app/hub.jl
"""

using Bonito

# Nav entries: route path + label. `/` is the default (interactive) app.
const _PLATFORM_TOOLS = [
    (path = "/",             title = "BRDF Analysis"),
    (path = "/rusinkiewicz", title = "Rusinkiewicz"),
    (path = "/sampling",     title = "Fit-Sampling"),
]

const _NAV_BG   = "#2c3e50"
const _NAV_LINK = "#ecf0f1"

"""
    _nav_bar(active_path) -> DOM

Slim top navigation bar. The link matching `active_path` is highlighted.
"""
function _nav_bar(active_path::String)
    function link(path, label)
        on_here = path == active_path
        DOM.a(label; href = path,
              style = "padding: 6px 12px; border-radius: 5px; text-decoration: none; " *
                      "font-size: 0.9em; white-space: nowrap; " *
                      (on_here ? "background: #3498db; color: white; font-weight: bold;" :
                                 "color: $(_NAV_LINK);"))
    end
    # The nav bar scrolls away with the page (it is not needed mid-session); the
    # interactive app makes its controls block sticky instead — see the controls
    # DOM.div in create_interactive_app.
    DOM.div(
        (link(t.path, t.title) for t in _PLATFORM_TOOLS)...,
        style = "display: flex; align-items: center; gap: 6px; flex-wrap: wrap; " *
                "padding: 8px 20px; background: $(_NAV_BG); " *
                "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;",
    )
end

"""
    run_platform(; port=8000, host="0.0.0.0", proxy_url=nothing)

Start the server with a top navigation bar on every page. `/` is the interactive
BRDF analysis app; the other tools get their own routes. This is the recommended
entry point; the single-tool `run_*_app` functions remain for serving one alone.

Routes:
- `/`              → BRDF analysis (default)
- `/rusinkiewicz`  → Rusinkiewicz parametrization
- `/sampling`      → fit-sampling
"""
function run_platform(; port = 8000, host = "0.0.0.0", proxy_url = nothing)
    purl = resolve_proxy_url(proxy_url, host, port)
    server = Bonito.Server(host, port; proxy_url = purl)

    # Wrap a tool page with the nav bar at the route level (create_* untouched).
    nav_wrap(path, create) = App(session -> DOM.div(_nav_bar(path), create(session)))

    Bonito.route!(server, "/"             => nav_wrap("/",             create_interactive_app))
    Bonito.route!(server, "/rusinkiewicz" => nav_wrap("/rusinkiewicz", create_rusinkiewicz_app))
    Bonito.route!(server, "/sampling"     => nav_wrap("/sampling",     create_sampling_app))

    println("=" ^ 66)
    println("BRDFToolkit")
    println("=" ^ 66)
    println("local:  http://$host:$port")
    hint = remote_url_hint(purl, port)
    isempty(hint) || println("remote: $hint")
    println("tools:  /  ·  /rusinkiewicz  ·  /sampling")
    println("$(Threads.nthreads()) thread(s)")
    println("Ctrl+C to stop")
    println("=" ^ 66)

    return server
end

export run_platform
