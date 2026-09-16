// renderer.js

import * as THREE from 'https://esm.sh/three@0.160.0';
import { OrbitControls } from 'https://esm.sh/three@0.160.0/examples/jsm/controls/OrbitControls.js';

// ── Base Vertex Shader (Shared across all models) ───────────────────────────
const BASE_VERTEX_SHADER = `
    varying vec3 vNormal;
    varying vec3 vWorldPosition;
    void main() {
        vec4 worldPos = modelMatrix * vec4(position, 1.0);
        vWorldPosition = worldPos.xyz;
        vNormal = normalize((modelMatrix * vec4(normal, 0.0)).xyz);
        gl_Position = projectionMatrix * viewMatrix * worldPos;
    }
`;

// ── Orbit-sync group ────────────────────────────────────────────────────────
// The three render panels (Measured / Fitted / Error) live under these fixed
// window[...] slots. When one panel's camera moves we mirror its pose onto the
// others so they rotate/zoom in lockstep. Reading window[key] (rather than a
// persistent renderer list) always targets the *current* renderer for each slot,
// so panels that get re-mounted on visibility toggles never leave stale entries.
const SYNC_KEYS = ['_brdfMeasured', '_brdfFitted', '_brdfError'];
// Module-level re-entrancy guard: applyCameraPose() calls controls.update(),
// which fires OrbitControls' 'change' event on the *target* renderer; this flag
// stops that echo from broadcasting back and looping forever.
let _orbitSyncing = false;

export function createUniversalRenderer(container, selfKey) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x202020);

    const initialW = container.clientWidth  || 400;
    const initialH = container.clientHeight || 400;

    const camera = new THREE.PerspectiveCamera(45, initialW / initialH, 0.1, 100);
    camera.position.set(0, 0, 4);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(initialW, initialH);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    // Sphere has radius 1; keep the camera outside it so zooming in doesn't put
    // the camera inside the mesh (every visible face would then be back-facing
    // and get discarded by the shader, blanking the sphere).
    controls.minDistance = 1.5;
    controls.maxDistance = 10;
    // OrbitControls' built-in wheel zoom proved unreliable here: zoomSpeed had no
    // visible effect and some mice jumped the whole min↔max range in one notch
    // (its per-event scale ignores deltaY magnitude). Drive the dolly ourselves
    // with a clamped, sign-aware per-event step so zoom is smooth on any mouse.
    controls.enableZoom = false;
    const ZOOM_STEP = 0.0015;  // fraction of distance per wheel unit (tune to taste)
    renderer.domElement.addEventListener('wheel', (event) => {
        event.preventDefault();
        const d = Math.max(-120, Math.min(120, event.deltaY)); // clamp runaway deltas
        const factor = Math.pow(1 + ZOOM_STEP, d);
        const offset = camera.position.clone().sub(controls.target);
        const dist = Math.min(controls.maxDistance,
                     Math.max(controls.minDistance, offset.length() * factor));
        offset.setLength(dist);
        camera.position.copy(controls.target).add(offset);
        broadcastCamera();   // custom wheel zoom bypasses OrbitControls' 'change'
    }, { passive: false });
    console.log('[brdf_renderer] init: custom wheel zoom active (step', ZOOM_STEP, ')');

    // ── Orbit synchronization ────────────────────────────────────────────────
    // Copy this panel's camera pose onto its peers. Called whenever the user
    // rotates/pans (OrbitControls 'change') or zooms (custom wheel handler).
    function broadcastCamera() {
        if (_orbitSyncing) return;
        _orbitSyncing = true;
        try {
            for (const key of SYNC_KEYS) {
                if (key === selfKey) continue;
                const peer = window[key];
                if (peer && peer.applyCameraPose) {
                    peer.applyCameraPose(camera.position, controls.target);
                }
            }
        } finally {
            _orbitSyncing = false;
        }
    }
    // Mirror a peer's pose onto this camera. controls.update() reconciles the
    // OrbitControls internal spherical state with the new position so the next
    // user interaction continues smoothly from here.
    function applyCameraPose(position, target) {
        camera.position.copy(position);
        controls.target.copy(target);
        controls.update();
    }
    controls.addEventListener('change', broadcastCamera);

    // Incident-direction marker: a distant "light bulb" (visual sun position)
    // plus a plain line from the origin out along wi (no arrowhead — a cone at
    // this scale always read as disproportionately thick), grouped so a single
    // toggle (setLightMarkerVisible) hides/shows both together.
    const lightOrbitRadius = 30.0;
    const wiLineLength = 3.0;
    const wiGroup = new THREE.Group();
    scene.add(wiGroup);
    const lightMesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.06, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0xffffaa })
    );
    wiGroup.add(lightMesh);
    const wiLineGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, wiLineLength),
    ]);
    const wiLine = new THREE.Line(wiLineGeom, new THREE.LineBasicMaterial({ color: 0xffcc33 }));
    wiGroup.add(wiLine);

    // ── Geometry factory ────────────────────────────────────────────────────
    // Each shape is sized to roughly fit a unit bounding radius so the existing
    // camera zoom clamp (minDistance 1.5) never ends up inside the mesh. Note the
    // arg meanings differ per geometry — e.g. TorusKnot's 2nd arg is the *tube*
    // radius, so naively reusing the sphere's (1, 64, 64) makes a tube of radius
    // 64 that swallows the camera and renders black (all faces back-facing).
    const GEOMETRY_FACTORIES = {
        sphere:      () => new THREE.SphereGeometry(1, 64, 64),
        cube:        () => new THREE.BoxGeometry(1.4, 1.4, 1.4),
        cylinder:    () => new THREE.CylinderGeometry(0.8, 0.8, 1.6, 64),
        cone:        () => new THREE.ConeGeometry(1, 1.8, 64),
        torus:       () => new THREE.TorusGeometry(0.75, 0.3, 32, 100),
        torusknot:   () => new THREE.TorusKnotGeometry(0.7, 0.25, 200, 32),
        icosahedron: () => new THREE.IcosahedronGeometry(1.1, 1),
    };
    let currentShape = 'sphere';

    // Sphere — starts with a placeholder material
    const sphere = new THREE.Mesh(
        GEOMETRY_FACTORIES[currentShape](),
        new THREE.MeshStandardMaterial({ color: 0x808080 })
    );
    scene.add(sphere);

    // Swap the mesh geometry in place, keeping the active material. Disposes the
    // old geometry so we don't leak GPU buffers on repeated switches.
    function setGeometry(name) {
        const factory = GEOMETRY_FACTORIES[name];
        if (!factory || name === currentShape) return;
        const old = sphere.geometry;
        sphere.geometry = factory();
        if (old) old.dispose();
        currentShape = name;
    }
    scene.add(new THREE.AmbientLight(0xffffff, 0.4));

    // Holds the currently loaded texture (MERL 3D Texture, or 2D LUT)
    let currentTexture = null;
    // Grid dims (N_th, N_td, N_pd, N_ph) of the universal LUT, captured at load
    // time so setMaterial can set uGridDims reliably — Bonito/msgpack may deliver
    // Julia arrays as array-likes that fail Array.isArray in the uniform parser.
    let currentGridDims = null;

    function setLightDir(theta, phi) {
        const x = Math.sin(theta) * Math.cos(phi);
        const y = Math.sin(theta) * Math.sin(phi);
        const z = Math.cos(theta);
        lightMesh.position.set(x * lightOrbitRadius, y * lightOrbitRadius, z * lightOrbitRadius);
        wiLine.geometry.setFromPoints([
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(x * wiLineLength, y * wiLineLength, z * wiLineLength),
        ]);
    }
    setLightDir(Math.PI / 4, 0);

    function setLightMarkerVisible(visible) {
        wiGroup.visible = visible;
    }

    async function loadTextureData(b64, config) {
        if (!b64 || !config) return;

        try {
            const bin = atob(b64);
            const bytes = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);

            if (true) {//(config.format === 'universal') {
                // 4D LUT folded into a 3D texture: depth = N_ph blocks of N_pd.
                // Wire payload is 3 channels/texel; expand to RGBA here.
                const [N_th, N_td, N_pd, N_ph] = config.dims;
                currentGridDims = [Number(N_th), Number(N_td), Number(N_pd), Number(N_ph)];
                const w = N_th, h = N_td, d = N_pd * N_ph;
                const n = w * h * d;
                const isFloat16 = (config.dtype === 'float16');

                // Reinterpret the raw little-endian bytes as the source typed array.
                const src = isFloat16
                    ? new Uint16Array(bytes.buffer, bytes.byteOffset, n * 3)
                    : new Float32Array(bytes.buffer, bytes.byteOffset, n * 3);

                if (isFloat16) {
                    const rgba = new Uint16Array(n * 4);
                    const ONE_H = 0x3c00; // half-float 1.0
                    for (let i = 0; i < n; i++) {
                        rgba[i * 4]     = src[i * 3];
                        rgba[i * 4 + 1] = src[i * 3 + 1];
                        rgba[i * 4 + 2] = src[i * 3 + 2];
                        rgba[i * 4 + 3] = ONE_H;
                    }
                    if (currentTexture) currentTexture.dispose();
                    currentTexture = new THREE.Data3DTexture(rgba, w, h, d);
                    currentTexture.type = THREE.HalfFloatType;
                } else {
                    const rgba = new Float32Array(n * 4);
                    for (let i = 0; i < n; i++) {
                        rgba[i * 4]     = src[i * 3];
                        rgba[i * 4 + 1] = src[i * 3 + 1];
                        rgba[i * 4 + 2] = src[i * 3 + 2];
                        rgba[i * 4 + 3] = 1.0;
                    }
                    if (currentTexture) currentTexture.dispose();
                    currentTexture = new THREE.Data3DTexture(rgba, w, h, d);
                    currentTexture.type = THREE.FloatType;
                }
                currentTexture.format = THREE.RGBAFormat;

            // } else if (config.format === 'merl') {
            //     // Dimensions: width (phi_d), height (theta_d), depth (theta_h)
            //     const [w, h, d] = config.dimensions;
            //     const n = w * h * d;
            //     const doubles = new Float64Array(bytes.buffer.slice(12)); // Skip 12-byte header
                
            //     const textureDataRGBA = new Float32Array(n * 4);
                
            //     for (let i = 0; i < n; i++) {
            //         textureDataRGBA[i * 4]     = doubles[i]         * config.scales[0];
            //         textureDataRGBA[i * 4 + 1] = doubles[i + n]     * config.scales[1];
            //         textureDataRGBA[i * 4 + 2] = doubles[i + 2 * n] * config.scales[2];
            //         textureDataRGBA[i * 4 + 3] = 1.0;
            //     }
                
            //     if (currentTexture) currentTexture.dispose();
            //     currentTexture = new THREE.Data3DTexture(textureDataRGBA, w, h, d);
            //     currentTexture.format = THREE.RGBAFormat;
            //     currentTexture.type = THREE.FloatType;
                
            } 

            if (currentTexture) {
                currentTexture.minFilter = THREE.LinearFilter;
                currentTexture.magFilter = THREE.LinearFilter;
                currentTexture.wrapS = THREE.ClampToEdgeWrapping;
                currentTexture.wrapT = THREE.ClampToEdgeWrapping;
                currentTexture.wrapR = THREE.ClampToEdgeWrapping;
                currentTexture.needsUpdate = true;
            }

            // Inject into the active material if it exists
            if (sphere.material && sphere.material.uniforms && sphere.material.uniforms.brdfData) {
                sphere.material.uniforms.brdfData.value = currentTexture;
                sphere.material.needsUpdate = true;
            }

        } catch (e) {
            console.error('[brdf_renderer] failed to load texture data:', e);
        }
    }

    // ── Material Factory ────────────────────────────────────────────────────
    function setMaterial(fragmentSnippet, uniformDefinitions) {
        const parsedUniforms = {};

        // Parse dictionary from Julia into THREE uniform objects.
        // Note: Bonito/msgpack may deliver Julia Vectors as array-LIKES (not true
        // Arrays), so test for indexable length rather than Array.isArray.
        const isVecLike = (v, len) =>
            v != null && typeof v === 'object' && v.length === len && typeof v[0] === 'number';
        for (const [key, val] of Object.entries(uniformDefinitions)) {
            if (isVecLike(val, 4)) {
                parsedUniforms[key] = { value: new THREE.Vector4(val[0], val[1], val[2], val[3]) };
            } else if (isVecLike(val, 3)) {
                parsedUniforms[key] = { value: new THREE.Vector3(val[0], val[1], val[2]) };
            } else if (typeof val === 'number') {
                parsedUniforms[key] = { value: val };
            }
        }

        // uGridDims is critical for the universal shader: if it stays at the
        // default vec4(0) the LUT lookup collapses to the corner (peak) texel and
        // the whole sphere renders the specular peak. Set it from the dims we
        // captured in loadTextureData so it never depends on uniform serialization.
        if (currentGridDims) {
            parsedUniforms.uGridDims = { value: new THREE.Vector4(
                currentGridDims[0], currentGridDims[1], currentGridDims[2], currentGridDims[3]) };
        }

        // Always inject required built-in uniforms
        parsedUniforms.lightPosition = { value: lightMesh.position };
        parsedUniforms.viewPosition = { value: camera.position };
        parsedUniforms.brdfData = { value: currentTexture };
        // Default to the lit path unless the payload explicitly asks for unlit.
        if (!('uUnlit' in parsedUniforms)) parsedUniforms.uUnlit = { value: 0.0 };

        // Construct the full fragment shader dynamically
        const fullFragmentShader = `
            precision highp float;
            precision highp sampler3D;
            precision highp sampler2D;

            uniform vec3 lightPosition;
            uniform vec3 viewPosition;
            // 1.0 = unlit false-color mode: the LUT holds display colors (e.g.
            // error-map output), shown as-is — no cosine/intensity shading and
            // no tone mapping, so the colormap stays readable. 0.0 = lit BRDF.
            uniform float uUnlit;

            // Universally injected by loadTextureData
            uniform sampler3D brdfData;

            varying vec3 vNormal;
            varying vec3 vWorldPosition;

            // --- Injected BRDF logic from Julia ---
            ${fragmentSnippet}
            // --------------------------------------

            void main() {
                vec3 N = normalize(vNormal);
                vec3 V = normalize(viewPosition - vWorldPosition);
                vec3 L = normalize(lightPosition - vWorldPosition);

                if (dot(N, V) < 0.0) discard;

                // Evaluate the universally named function
                vec3 color = brdf_evaluate(L, V, N);

                // Simple tone mapping to prevent blowout (skipped in unlit mode:
                // Reinhard is per-channel nonlinear and would shift colormap hues)
                color = mix(color / (color + vec3(1.0)), color, uUnlit);

                gl_FragColor = vec4(color, 1.0);
            }
        `;

        const newMaterial = new THREE.ShaderMaterial({
            uniforms: parsedUniforms,
            vertexShader: BASE_VERTEX_SHADER,
            fragmentShader: fullFragmentShader
        });

        if (sphere.material) sphere.material.dispose();
        sphere.material = newMaterial;
    }

    // ── Uniform Updater ─────────────────────────────────────────────────────
    function updateUniforms(newValues) {
        if (!sphere.material || !sphere.material.uniforms) return;
        
        for (const [key, val] of Object.entries(newValues)) {
            if (sphere.material.uniforms[key]) {
                if (Array.isArray(val) && val.length === 3) {
                    sphere.material.uniforms[key].value.set(val[0], val[1], val[2]);
                } else if (typeof val === 'number') {
                    sphere.material.uniforms[key].value = val;
                }
            }
        }
    }

    // ── Render Loop ─────────────────────────────────────────────────────────
    let raf = 0;
    function animate() {
        raf = requestAnimationFrame(animate);
        if (sphere.material && sphere.material.uniforms) {
            if (sphere.material.uniforms.lightPosition) {
                sphere.material.uniforms.lightPosition.value.copy(lightMesh.position);
            }
            if (sphere.material.uniforms.viewPosition) {
                sphere.material.uniforms.viewPosition.value.copy(camera.position);
            }
        }
        controls.update();
        renderer.render(scene, camera);
    }
    animate();

    // ── Housekeeping ────────────────────────────────────────────────────────
    function resize() {
        const w = container.clientWidth  || 400;
        const h = container.clientHeight || 400;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    }
    const ro = (typeof ResizeObserver !== 'undefined') ? new ResizeObserver(resize) : null;
    if (ro) ro.observe(container);

    function dispose() {
        cancelAnimationFrame(raf);
        if (ro) ro.disconnect();
        if (sphere.material) sphere.material.dispose();
        sphere.geometry.dispose();
        if (currentTexture) currentTexture.dispose();
        renderer.dispose();
        if (renderer.domElement.parentNode === container) {
            container.removeChild(renderer.domElement);
        }
    }

    return { setLightDir, setLightMarkerVisible, setGeometry, loadTextureData, setMaterial, updateUniforms, applyCameraPose, resize, dispose };
}