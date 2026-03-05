#!/usr/bin/env julia

"""
Startup script for BRDF Analysis Platform - Interactive Version

This version includes:
- File upload functionality
- WGLMakie hemisphere visualization
- Interactive angle controls

Usage:
    julia --project=. start_interactive.jl
    julia --project=. start_interactive.jl --port 8080
"""
# Activate the project environment
using Pkg
Pkg.activate(".")
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
println("\n🎨 Welcome to BRDF Analysis Platform (Interactive)")
println("📚 Features: File Upload + WGLMakie Visualization")
println("📖 For help, see: UI_GUIDE.md or QUICKSTART.md\n")

try
    server = run_interactive_app(port=port, host=host)

    # Keep the server running
    println("\n✨ Server is running. Waiting for connections...")
    println("📋 Instructions:")
    println("   1. Open http://$host:$port in your browser")
    println("   2. Click 'Choose File' to load a .binary BRDF file")
    println("   3. Adjust incident angles with the sliders")
    println("   4. Watch the hemisphere plot update in real-time!")
    println()

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
