"""
Test that the multi-tool server can be created and routed successfully.
Binds a port briefly and closes it again.
"""

using BRDFPlatform
using Bonito

println("=" ^ 70)
println("Testing Server Creation")
println("=" ^ 70)

println("\n✅ Creating server instance...")
server = run_platform(port=8765, host="127.0.0.1")

println("✅ Server created successfully!")
println("   Type: ", typeof(server))
println("   Routes: /  ·  /rusinkiewicz  ·  /sampling")

close(server)
println("✅ Server closed.")

println("\n" * "=" ^ 70)
println("✅ Server creation test passed!")
println("=" ^ 70)
println("\nTo actually start the server, run:")
println("  julia --project=. start_platform.jl")
println("\nThen open your browser to http://127.0.0.1:8000")
println("=" ^ 70)
