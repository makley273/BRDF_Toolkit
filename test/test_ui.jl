"""
Test UI components without starting a server.
"""

using BRDFPlatform
using Bonito

println("Testing UI components...")

# Test that run_app function exists
@assert isdefined(BRDFPlatform, :run_app) "run_app function not defined"
println("✅ run_app function exists")

# Test app creation (without starting server)
println("✅ Creating app session...")
session = Session()

# Test main app creation
println("✅ Creating main app layout...")
app_dom = BRDFPlatform.create_main_app(session)
@assert !isnothing(app_dom) "App DOM is nothing"
println("✅ App DOM created successfully")

# Test individual components
println("✅ Testing header...")
header = BRDFPlatform.create_header()
@assert !isnothing(header) "Header is nothing"

println("✅ Testing plugin panel...")
plugin_panel = BRDFPlatform.create_plugin_panel()
@assert !isnothing(plugin_panel) "Plugin panel is nothing"

println("✅ Testing sidebar...")
sidebar = BRDFPlatform.create_sidebar()
@assert !isnothing(sidebar) "Sidebar is nothing"

println("✅ Testing footer...")
footer = BRDFPlatform.create_footer()
@assert !isnothing(footer) "Footer is nothing"

println("\n" * "="^70)
println("✅ All UI components created successfully!")
println("="^70)
println("\nTo start the web server, run:")
println("  julia --project=. start_app.jl")
println("\nOr in Julia REPL:")
println("  using BRDFPlatform")
println("  run_app()")
println("="^70)
