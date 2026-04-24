# test/glsl_validation.jl
#
# Universal GLSL ↔ Julia validation for all BRDF model plugins.
#
# For every registered model a 64×64 hemisphere UV grid is rendered via Three.js
# WebGL into a float render target.  The resulting pixel buffer is compared
# channel-by-channel against Julia evaluate() on the identical grid.
#
# Tolerances (per CLAUDE.md):
#   LambertianModel  — max absolute error < 1e-4  (algebraically exact)
#   Microfacet models — max absolute error < 1e-3  (trig / fp32 rounding)
#
# Running:
#   julia --project=. test/glsl_validation.jl
#
# The test starts a Bonito server on port 9990.  Open the URL in any browser
# (Chrome or Firefox recommended) to trigger the WebGL renders.  Results are
# reported in the Julia REPL.
# For fully automated / CI use, connect via Playwright.jl instead.

using Test
using Bonito
using StaticArrays
using Colors
using LinearAlgebra

using BRDFPlatform

const THREE = Bonito.ES6Module("https://unpkg.com/three@0.160.0/build/three.module.js")

# ============================================================================
# Configuration
# ============================================================================

# 64×64 = 4 096 samples — covers the hemisphere well, runs in a few seconds
const RESOLUTION = 64

# Three outgoing directions per model: normal incidence, 30°, 60° off-normal
const WO_DIRS = (
    normalize(SVector{3,Float32}(0f0,    0f0,   1f0   )),
    normalize(SVector{3,Float32}(0.5f0,  0f0,   0.866f0)),
    normalize(SVector{3,Float32}(0.866f0, 0f0,  0.5f0 )),
)

const N_VEC = SVector{3,Float32}(0f0, 0f0, 1f0)

# ============================================================================
# Test instances — one representative set of parameters per model type
# ============================================================================

const TEST_MODELS = (
    LambertianModel(RGB{Float32}(0.7f0, 0.2f0, 0.1f0)),

    CookTorranceModel(
        0.4f0,
        RGB{Float32}(0.04f0, 0.06f0, 0.08f0),
        RGB{Float32}(0.5f0,  0.3f0,  0.1f0 ),
    ),
    CookTorranceSpecularModel(
        0.3f0,
        RGB{Float32}(0.8f0, 0.6f0, 0.1f0),
    ),
    CookTorranceFresnelDiffuseModel(
        0.5f0,
        RGB{Float32}(0.04f0, 0.04f0, 0.04f0),
        RGB{Float32}(0.6f0,  0.4f0,  0.2f0 ),
    ),
    CookTorranceScaledDiffuseModel(
        0.35f0,
        RGB{Float32}(0.05f0, 0.05f0, 0.05f0),
        RGB{Float32}(0.4f0,  0.5f0,  0.6f0 ),
        0.6f0,
    ),

    GGXModel(
        0.4f0,
        RGB{Float32}(0.04f0, 0.06f0, 0.08f0),
        RGB{Float32}(0.5f0,  0.3f0,  0.1f0 ),
    ),
    GGXSpecularModel(
        0.3f0,
        RGB{Float32}(0.8f0, 0.6f0, 0.1f0),
    ),
    GGXFresnelDiffuseModel(
        0.5f0,
        RGB{Float32}(0.04f0, 0.04f0, 0.04f0),
        RGB{Float32}(0.6f0,  0.4f0,  0.2f0 ),
    ),
    GGXScaledDiffuseModel(
        0.35f0,
        RGB{Float32}(0.05f0, 0.05f0, 0.05f0),
        RGB{Float32}(0.4f0,  0.5f0,  0.6f0 ),
        0.6f0,
    ),
)

# ============================================================================
# Tolerances
# ============================================================================

glsl_tolerance(::LambertianModel) = 1f-4
glsl_tolerance(::BRDFModel)       = 1f-3

# ============================================================================
# Julia reference grid
# ============================================================================

"""
    julia_reference(model, resolution, wo, N) -> Vector{Float32}

Evaluate `model` over a `resolution × resolution` UV hemisphere grid using
the same mapping as the GLSL wrapper shader.  Returns an RGBA float buffer
in row-major order matching `readRenderTargetPixels`.
"""
function julia_reference(
    model,
    resolution::Int,
    wo::SVector{3,Float32},
    N::SVector{3,Float32},
)::Vector{Float32}
    buf = Vector{Float32}(undef, resolution * resolution * 4)
    idx = 1
    for y in 0:resolution-1
        for x in 0:resolution-1
            u     = (Float32(x) + 0.5f0) / Float32(resolution)
            v     = (Float32(y) + 0.5f0) / Float32(resolution)
            theta = u * (Float32(π) / 2f0)
            phi   = v * (2f0 * Float32(π))
            wi = SVector{3,Float32}(
                sin(theta) * cos(phi),
                sin(theta) * sin(phi),
                cos(theta),
            )
            c = evaluate(model, wi, wo)
            buf[idx]   = c.r
            buf[idx+1] = c.g
            buf[idx+2] = c.b
            buf[idx+3] = 1f0
            idx += 4
        end
    end
    return buf
end

# ============================================================================
# GLSL render scheduler
# ============================================================================

"""
    schedule_glsl_render!(session, model, resolution, wo, N, result_obs)

Fire a single offscreen Three.js render for `model` and store the resulting
RGBA float buffer in `result_obs` once the GPU readback completes.

Design notes:
- `shader_parameters(model)` is serialised to a JS object via Bonito's `\$(...)`.
  Three.js accepts plain JS arrays for vecN uniforms (uses `gl.uniform3fv`
  internally), so no explicit `THREE.Vector3` construction is needed.
- `glsl_code` contains the model's uniform declarations plus `brdf_evaluate`.
  The wrapper adds `u_wo` / `u_N` and the `main()` entry point.
- All GLSL snippets define `brdf_evaluate(vec3 wi, vec3 wo, vec3 N) → vec3`,
  which is the shared contract every model shader must satisfy.
"""
function schedule_glsl_render!(
    session,
    model,
    resolution::Int,
    wo::SVector{3,Float32},
    N::SVector{3,Float32},
    result_obs::Observable{Vector{Float32}},
)
    glsl_code, _ = glsl_shader(model)
    params        = shader_parameters(model)   # Dict{String, Any}

    evaljs(session, js"""
        (async () => {
            const THREE = await import($(THREE));
            const res   = $(resolution);

            const renderer = new THREE.WebGLRenderer({ antialias: false });
            const target   = new THREE.WebGLRenderTarget(res, res, {
                type:   THREE.FloatType,
                format: THREE.RGBAFormat,
            });

            const scene  = new THREE.Scene();
            const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

            // Build Three.js uniforms from the Julia shader_parameters dict.
            // Three.js accepts plain JS arrays for vecN uniforms via uniform3fv,
            // so no THREE.Vector3 wrapping is required.
            const raw = $(params);
            const uniforms = {};
            for (const [k, v] of Object.entries(raw)) {
                uniforms[k] = { value: v };
            }
            // Geometry uniforms added by the test wrapper (not part of the model)
            uniforms["u_wo"] = { value: [$(wo[1]), $(wo[2]), $(wo[3])] };
            uniforms["u_N"]  = { value: [$(N[1]),  $(N[2]),  $(N[3])] };

            const material = new THREE.ShaderMaterial({
                uniforms,
                vertexShader: `
                    void main() { gl_Position = vec4(position, 1.0); }
                `,
                fragmentShader: `
                    uniform vec3 u_wo;
                    uniform vec3 u_N;

                    $(glsl_code)

                    void main() {
                        // Same UV → direction mapping as julia_reference()
                        vec2  uv    = gl_FragCoord.xy / float($(resolution));
                        float theta = uv.x * 1.5707963268;   // u → [0, π/2]
                        float phi   = uv.y * 6.2831853072;   // v → [0, 2π]
                        vec3  wi    = vec3(sin(theta) * cos(phi),
                                           sin(theta) * sin(phi),
                                           cos(theta));
                        gl_FragColor = vec4(brdf_evaluate(wi, u_wo, u_N), 1.0);
                    }
                `,
            });

            scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2.0, 2.0), material));

            renderer.setRenderTarget(target);
            renderer.render(scene, camera);

            const buf = new Float32Array(res * res * 4);
            renderer.readRenderTargetPixels(target, 0, 0, res, res, buf);

            $(result_obs).notify(Array.from(buf));
        })().catch(err => console.error("[BRDF validation]", err));
    """)
end

# ============================================================================
# Test suite
# ============================================================================

@testset "GLSL vs Julia BRDF validation" begin

    # Pre-allocate one Observable per (model, wo direction) combination
    obs_table = Dict{Tuple{Int,Int}, Observable{Vector{Float32}}}(
        (mi, wi) => Observable{Vector{Float32}}(Float32[])
        for mi in eachindex(TEST_MODELS), wi in eachindex(WO_DIRS)
    )

    # All renders are scheduled in a single browser session to minimise overhead.
    # The App block runs once when the first client connects.
    app = App() do session
        for mi in eachindex(TEST_MODELS), wi in eachindex(WO_DIRS)
            schedule_glsl_render!(
                session,
                TEST_MODELS[mi], RESOLUTION,
                WO_DIRS[wi], N_VEC,
                obs_table[(mi, wi)],
            )
        end
        return DOM.div("BRDF GLSL validation running — see Julia REPL for results.")
    end

    server = Bonito.Server(app, "127.0.0.1", 9990)
    @info "Open http://127.0.0.1:9990 to trigger GLSL renders"

    try
        for (mi, model) in enumerate(TEST_MODELS)
            model_name = nameof(typeof(model))
            tol        = glsl_tolerance(model)

            @testset "$model_name" begin
                for (wi, wo) in enumerate(WO_DIRS)
                    obs = obs_table[(mi, wi)]

                    # Wait for the GPU readback with a timeout
                    timeout_s = 60.0
                    t0 = time()
                    while isempty(obs[])
                        sleep(0.05)
                        if time() - t0 > timeout_s
                            error("Timeout waiting for GLSL result: $model_name wo_$wi")
                        end
                    end

                    julia_buf = julia_reference(model, RESOLUTION, wo, N_VEC)
                    glsl_buf  = obs[]

                    @assert length(glsl_buf) == length(julia_buf) "Buffer length mismatch"

                    diff     = abs.(glsl_buf .- julia_buf)
                    max_err  = maximum(diff)
                    mean_err = sum(diff) / length(diff)

                    @test max_err < tol

                    status = max_err < tol ? "✓" : "✗"
                    @info "$status $model_name  wo_$wi  " *
                          "max=$(round(max_err, sigdigits=3))  " *
                          "mean=$(round(mean_err, sigdigits=3))  " *
                          "tol=$tol"
                end
            end
        end
    finally
        close(server)
    end
end
