"""
Test that the server can be created successfully.
Does not actually start the server (which would block).
"""

using BRDFPlatform
using Bonito

println("=" ^ 70)
println("Testing Server Creation")
println("=" ^ 70)

println("\n✅ Creating server instance...")
server = run_app(port=8765, host="127.0.0.1")

println("✅ Server created successfully!")
println("   Type: ", typeof(server))
println("   Server would be available at: http://127.0.0.1:8765")

println("\n" * "=" ^ 70)
println("✅ Server creation test passed!")
println("=" ^ 70)
println("\nTo actually start the server, run:")
println("  julia --project=. start_app.jl")
println("\nThen open your browser to http://127.0.0.1:8000")
println("=" ^ 70)
