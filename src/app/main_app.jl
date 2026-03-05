"""
Main Bonito.jl Web Application

Interactive web interface for BRDF analysis platform.
Provides file upload, format detection, model fitting, and visualization.
"""

using Bonito
using WGLMakie

# Types are in scope from BRDFPlatform module

"""
    create_header() -> DOM

Create the application header with title and info.
"""
function create_header()
    return DOM.div(
        DOM.h1("🔬 BRDF Analysis Platform", style="color: #2c3e50; margin-bottom: 10px;"),
        DOM.p("Interactive tool for analyzing and fitting BRDF models to measured material data",
              style="color: #7f8c8d; margin-bottom: 20px;"),
        DOM.hr(style="border: 1px solid #ecf0f1; margin-bottom: 20px;")
    )
end

"""
    create_plugin_panel() -> DOM

Create a panel showing registered plugins.
"""
function create_plugin_panel()
    plugins = list_plugins()

    format_count = length(plugins[:formats])
    model_count = length(plugins[:models])
    optimizer_count = length(plugins[:optimizers])

    return DOM.div(
        DOM.h3("📦 Loaded Plugins", style="color: #34495e;"),
        DOM.div(
            DOM.div(
                DOM.strong("Formats: "),
                DOM.span("$format_count", style="color: #27ae60; font-weight: bold;"),
                DOM.span(" ($(join(plugins[:formats], ", ")))",
                         style="color: #7f8c8d; font-size: 0.9em;")
            ),
            DOM.div(
                DOM.strong("Models: "),
                DOM.span("$model_count", style="color: #e67e22; font-weight: bold;"),
                DOM.span(model_count > 0 ? " ($(join(plugins[:models], ", ")))" : " (none yet)",
                         style="color: #7f8c8d; font-size: 0.9em;")
            ),
            DOM.div(
                DOM.strong("Optimizers: "),
                DOM.span("$optimizer_count", style="color: #3498db; font-weight: bold;"),
                DOM.span(optimizer_count > 0 ? " ($(join(plugins[:optimizers], ", ")))" : " (none yet)",
                         style="color: #7f8c8d; font-size: 0.9em;")
            ),
            style="padding: 15px; background: #ecf0f1; border-radius: 5px; margin-bottom: 20px;"
        )
    )
end

"""
    create_file_upload_panel(session) -> DOM

Create file upload interface with drag-and-drop support.
"""
function create_file_upload_panel(session)
    # Observable to store uploaded file info
    file_info = Observable(Dict{String, Any}("status" => "no_file"))

    # Create file input
    file_input = DOM.input(
        type="file",
        accept=".binary,.bird,.json",
        id="brdf-file-input",
        style="margin: 10px 0;"
    )

    # Status display
    status_display = map(file_info) do info
        if info["status"] == "no_file"
            return DOM.div(
                "📁 No file loaded. Please select a BRDF file (.binary, .bird, .json)",
                style="color: #7f8c8d; padding: 10px; font-style: italic;"
            )
        elseif info["status"] == "loading"
            return DOM.div(
                "⏳ Loading file: $(info["name"])...",
                style="color: #3498db; padding: 10px; font-weight: bold;"
            )
        elseif info["status"] == "loaded"
            return DOM.div(
                DOM.div("✅ Successfully loaded: $(info["name"])",
                       style="color: #27ae60; font-weight: bold;"),
                DOM.div("Format: $(info["format"])", style="margin-top: 5px;"),
                DOM.div("Samples: $(info["samples"])", style="margin-top: 5px;"),
                DOM.div("Wavelengths: $(info["wavelengths"]) nm", style="margin-top: 5px;"),
                style="padding: 10px; background: #d5f4e6; border-left: 4px solid #27ae60; border-radius: 3px;"
            )
        elseif info["status"] == "error"
            return DOM.div(
                DOM.div("❌ Error loading file: $(info["name"])",
                       style="color: #e74c3c; font-weight: bold;"),
                DOM.div("$(info["error"])",
                       style="margin-top: 5px; font-size: 0.9em;"),
                style="padding: 10px; background: #fadbd8; border-left: 4px solid #e74c3c; border-radius: 3px;"
            )
        end
    end

    return DOM.div(
        DOM.h3("📂 Load BRDF Data", style="color: #34495e;"),
        DOM.div(
            file_input,
            status_display,
            style="padding: 15px; border: 2px dashed #bdc3c7; border-radius: 5px; margin-bottom: 20px;"
        )
    )
end

"""
    create_visualization_panel(session) -> DOM

Create placeholder for BRDF visualization (WGLMakie plots, Three.js rendering).
"""
function create_visualization_panel(session)
    return DOM.div(
        DOM.h3("📊 Visualization", style="color: #34495e;"),
        DOM.div(
            DOM.p("Visualization panel - will display:"),
            DOM.ul(
                DOM.li("Hemisphere plots (WGLMakie)"),
                DOM.li("Error heatmaps"),
                DOM.li("3D material rendering (Three.js)"),
                DOM.li("Model comparison views")
            ),
            DOM.p("Coming in Phase 2!",
                  style="color: #e67e22; font-weight: bold; margin-top: 10px;"),
            style="padding: 15px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; color: #6c757d;"
        )
    )
end

"""
    create_model_fitting_panel(session) -> DOM

Create panel for model selection and fitting.
"""
function create_model_fitting_panel(session)
    return DOM.div(
        DOM.h3("🎯 Model Fitting", style="color: #34495e;"),
        DOM.div(
            DOM.p("Model fitting controls - will include:"),
            DOM.ul(
                DOM.li("Model selection dropdown"),
                DOM.li("Optimizer selection"),
                DOM.li("Fit button"),
                DOM.li("Parameter display"),
                DOM.li("Error metrics")
            ),
            DOM.p("Coming in Phase 2!",
                  style="color: #e67e22; font-weight: bold; margin-top: 10px;"),
            style="padding: 15px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; color: #6c757d;"
        )
    )
end

"""
    create_footer() -> DOM

Create application footer with info and links.
"""
function create_footer()
    return DOM.div(
        DOM.hr(style="border: 1px solid #ecf0f1; margin-top: 30px; margin-bottom: 15px;"),
        DOM.div(
            DOM.p(
                "BRDF Analysis Platform • Bachelor Thesis Project • Maria Alexandra Kley • 2025",
                style="text-align: center; color: #95a5a6; font-size: 0.9em;"
            ),
            DOM.p(
                DOM.a("Documentation", href="#", style="color: #3498db; margin: 0 10px;"),
                DOM.a("GitHub", href="#", style="color: #3498db; margin: 0 10px;"),
                DOM.a("Report Issue", href="#", style="color: #3498db; margin: 0 10px;"),
                style="text-align: center; color: #95a5a6; font-size: 0.85em;"
            )
        )
    )
end

"""
    create_sidebar() -> DOM

Create a sidebar with quick actions and settings.
"""
function create_sidebar()
    return DOM.div(
        DOM.h4("⚙️ Quick Actions", style="color: #34495e; margin-top: 0;"),
        DOM.button(
            "🔄 Reload User Plugins",
            onclick=js"""
                alert('Reloading user plugins...');
                // TODO: Implement plugin reload via Julia callback
            """,
            style="width: 100%; padding: 10px; margin-bottom: 10px; background: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer;"
        ),
        DOM.button(
            "📊 Export Results",
            onclick=js"alert('Export functionality coming soon!')",
            style="width: 100%; padding: 10px; margin-bottom: 10px; background: #95a5a6; color: white; border: none; border-radius: 5px; cursor: pointer;",
            disabled=true
        ),
        DOM.button(
            "ℹ️ Help",
            onclick=js"alert('Help documentation coming soon!')",
            style="width: 100%; padding: 10px; background: #2ecc71; color: white; border: none; border-radius: 5px; cursor: pointer;"
        ),
        DOM.hr(style="margin: 20px 0;"),
        DOM.div(
            DOM.h4("📖 Quick Guide", style="color: #34495e;"),
            DOM.ol(
                DOM.li("Load BRDF data file", style="margin: 5px 0; font-size: 0.9em;"),
                DOM.li("Select BRDF model", style="margin: 5px 0; font-size: 0.9em;"),
                DOM.li("Choose optimizer", style="margin: 5px 0; font-size: 0.9em;"),
                DOM.li("Fit model to data", style="margin: 5px 0; font-size: 0.9em;"),
                DOM.li("Compare results", style="margin: 5px 0; font-size: 0.9em;")
            ),
            style="color: #7f8c8d; font-size: 0.85em;"
        ),
        style="width: 250px; padding: 20px; background: #f8f9fa; border-right: 1px solid #dee2e6; min-height: 100vh;"
    )
end

"""
    create_main_app(session) -> DOM

Create the main application layout.
"""
function create_main_app(session)
    # Main content area
    main_content = DOM.div(
        create_header(),
        create_plugin_panel(),
        create_file_upload_panel(session),
        DOM.div(
            # Two-column layout for model fitting and visualization
            DOM.div(
                create_model_fitting_panel(session),
                style="flex: 1; margin-right: 10px;"
            ),
            DOM.div(
                create_visualization_panel(session),
                style="flex: 1; margin-left: 10px;"
            ),
            style="display: flex; gap: 20px;"
        ),
        create_footer(),
        style="flex: 1; padding: 30px; max-width: 1400px; margin: 0 auto;"
    )

    # Full layout with sidebar
    return DOM.div(
        create_sidebar(),
        main_content,
        style="display: flex; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;"
    )
end

"""
    run_app(; port=8000, host="127.0.0.1")

Start the Bonito web application.

# Arguments
- `port`: Port number (default: 8000)
- `host`: Host address (default: "127.0.0.1")
"""
function run_app(; port=8000, host="127.0.0.1")
    app = App() do session
        return create_main_app(session)
    end

    println("=" ^ 70)
    println("🚀 Starting BRDF Analysis Platform")
    println("=" ^ 70)
    println("📡 Server running at: http://$host:$port")
    println("👉 Open your browser and navigate to the URL above")
    println("⏹️  Press Ctrl+C to stop the server")
    println("=" ^ 70)

    # Start the server (Bonito uses positional args: app, host, port)
    server = Bonito.Server(app, host, port)

    return server
end

# Export the run function
export run_app
