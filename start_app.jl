#!/usr/bin/env julia

"""
Startup script for BRDF Analysis Platform web application.

Usage:
    julia --project=. start_app.jl
    julia --project=. start_app.jl --port 8080
    julia --project=. start_app.jl --port 8080 --host 0.0.0.0
"""

using BRDFPlatform

# Parse command line arguments
port = 8000
host = "127.0.0.1"

for (i, arg) in enumerate(ARGS)
    if arg == "--port" && i < length(ARGS)
        global port = parse(Int, ARGS[i + 1])
    elseif arg == "--host" && i < length(ARGS)
        global host = ARGS[i + 1]
    end
end

# Start the application
println("\n🎨 Welcome to BRDF Analysis Platform")
println("📚 For help, see: README.md or CLAUDE.MD\n")

try
    server = run_app(port=port, host=host)

    # Keep the server running
    println("\n✨ Server is running. Waiting for connections...\n")
    wait(server)
catch e
    if isa(e, InterruptException)
        println("\n\n👋 Shutting down server...")
        println("Goodbye!\n")
    else
        println("\n❌ Error starting server:")
        showerror(stdout, e, catch_backtrace())
        println()
    end
end
