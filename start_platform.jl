#!/usr/bin/env julia

"""
Startup script for BRDFToolkit

Serves every tool from a single server with a top navigation bar:
- /                → BRDF analysis (upload, fit, visualize) — the default
- /rusinkiewicz    → Rusinkiewicz parametrization
- /sampling        → fit-sampling

Usage:
    julia --project=. start_platform.jl
    julia --project=. start_platform.jl --port 8080
"""
# Activate the project environment
using Pkg
Pkg.activate(".")
using BRDFPlatform

# Parse command line arguments
port = 8000
host = "0.0.0.0"

for (i, arg) in enumerate(ARGS)
    if arg == "--port" && i < length(ARGS)
        global port = parse(Int, ARGS[i + 1])
    elseif arg == "--host" && i < length(ARGS)
        global host = ARGS[i + 1]
    end
end

# Start the application
println("\nWelcome to BRDFToolkit")
println("Tools: BRDF Analysis · Rusinkiewicz · Fit-Sampling")
println("For help, see: README.md\n")

try
    server = run_platform(port=port, host=host)

    # Keep the server running
    println("\nServer is running. Waiting for connections...")
    println("Instructions:")
    println("   1. Open http://$host:$port in your browser")
    println("   2. Pick a tool from the menu (or use the top navigation bar)")
    println("   3. Switch between tools anytime — they share one server")
    println()

    wait(server)
catch e
    if isa(e, InterruptException)
        println("\n\nShutting down server...")
        println("Goodbye!\n")
    else
        println("\nError starting server:")
        showerror(stdout, e, catch_backtrace())
        println()
    end
end
