// merl_renderer.js
//
// Three.js renderer for MEASURED MERL BRDF data, exposed as an ES6 module
// to Bonito.jl. The MERL material class is inlined here (originally
// MERLBRDFMaterial_new.js) so this file is self-contained — Bonito's
// ES6Module loader does not resolve relative imports between local files.
//
// Public API:
//   createMerlRenderer(container) -> { setLightDir, loadBufferBase64,
//                                      resize, dispose }

// Use esm.sh: it rewrites bare `import 'three'` inside OrbitControls so the
// bundle step (deno bundle, run by Bonito) can resolve all dependencies.
import * as THREE from 'https://esm.sh/three@0.160.0';
import { OrbitControls } from 'https://esm.sh/three@0.160.0/examples/jsm/controls/OrbitControls.js';

// ── MERL constants (identical to MERLBRDFMaterial_new.js) ────────────────────
const BRDF_SAMPLING_RES_THETA_H = 90;
const BRDF_SAMPLING_RES_THETA_D = 90;
const BRDF_SAMPLING_RES_PHI_D   = 360;
const RED_SCALE   = 1.0  / 1500.0;
const GREEN_SCALE = 1.15 / 1500.0;
const BLUE_SCALE  = 1.66 / 1500.0;

// ── MERL ShaderMaterial (verbatim from MERLBRDFMaterial_new.js) ──────────────
class MERLBRDFMaterial extends THREE.ShaderMaterial {
    static async loadBRDFFromBuffer(buffer) {
        const dims = new Int32Array(buffer, 0, 3);
        const n = dims[0] * dims[1] * dims[2];
        const expectedN = BRDF_SAMPLING_RES_THETA_H *
                          BRDF_SAMPLING_RES_THETA_D *
                          Math.floor(BRDF_SAMPLING_RES_PHI_D / 2);
        if (n !== expectedN) {
            throw new Error('BRDF dimensions mismatch');
        }

        const doubles = new Float64Array(buffer.slice(12));

        const width  = Math.floor(BRDF_SAMPLING_RES_PHI_D / 2);
        const height = BRDF_SAMPLING_RES_THETA_D;
        const depth  = BRDF_SAMPLING_RES_THETA_H;
        const data = new Float32Array(width * height * depth * 3);

        for (let i = 0; i < n; i++) {
            data[i * 3]     = doubles[i]         * RED_SCALE;
            data[i * 3 + 1] = doubles[i + n]     * GREEN_SCALE;
            data[i * 3 + 2] = doubles[i + 2 * n] * BLUE_SCALE;
        }

        const dataRGBA = new Float32Array(width * height * depth * 4);
        for (let i = 0; i < width * height * depth; i++) {
            dataRGBA[i * 4]     = data[i * 3];
            dataRGBA[i * 4 + 1] = data[i * 3 + 1];
            dataRGBA[i * 4 + 2] = data[i * 3 + 2];
            dataRGBA[i * 4 + 3] = 1.0;
        }

        const texture = new THREE.Data3DTexture(dataRGBA, width, height, depth);
        texture.format    = THREE.RGBAFormat;
        texture.type      = THREE.FloatType;
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.wrapS     = THREE.ClampToEdgeWrapping;
        texture.wrapT     = THREE.ClampToEdgeWrapping;
        texture.wrapR     = THREE.ClampToEdgeWrapping;
        texture.needsUpdate = true;
        return texture;
    }

    constructor(brdfTexture) {
        const vertexShader = `
            varying vec3 vNormal;
            varying vec3 vWorldPosition;
            void main() {
                vec4 worldPos = modelMatrix * vec4(position, 1.0);
                vWorldPosition = worldPos.xyz;
                vNormal = normalize((modelMatrix * vec4(normal, 0.0)).xyz);
                gl_Position = projectionMatrix * viewMatrix * worldPos;
            }
        `;

        const fragmentShader = `
            precision highp float;
            precision highp sampler3D;

            uniform sampler3D brdfData;
            uniform vec3 lightPosition;
            uniform vec3 viewPosition;

            varying vec3 vNormal;
            varying vec3 vWorldPosition;

            const float M_PI = 3.14159265359;

            float theta_half_index(float theta_half) {
                if (theta_half <= 0.0) return 0.0;
                float theta_half_deg = (theta_half / (M_PI/2.0)) * 90.0;
                float temp = sqrt(theta_half_deg * 90.0);
                return clamp(temp, 0.0, 89.0);
            }
            float theta_diff_index(float theta_diff) {
                return clamp(theta_diff / (M_PI * 0.5) * 90.0, 0.0, 89.0);
            }
            float phi_diff_index(float phi_diff) {
                if (phi_diff < 0.0) phi_diff += M_PI;
                return clamp(phi_diff / M_PI * 180.0, 0.0, 179.0);
            }

            void main() {
                vec3 N = normalize(vNormal);
                vec3 V = normalize(viewPosition - vWorldPosition);
                vec3 L = normalize(lightPosition - vWorldPosition);
                if (dot(N, V) < 0.0) discard;

                vec3 H = normalize(L + V);
                float NdotL = clamp(dot(N, L), 0.0, 1.0);
                float NdotH = clamp(dot(N, H), 0.0, 1.0);
                float LdotH = clamp(dot(L, H), 0.0, 1.0);

                float theta_half = acos(NdotH);
                float theta_diff = acos(LdotH);

                float sin_theta_h = sqrt(max(1.0 - NdotH * NdotH, 0.0));
                float sin_theta_d = sqrt(max(1.0 - LdotH * LdotH, 0.0));
                float numerator   = NdotL - NdotH * LdotH;
                float denominator = sin_theta_h * sin_theta_d;
                float cos_phi_d   = numerator / (denominator + 1e-7);
                float phi_diff    = acos(clamp(cos_phi_d, -1.0, 1.0));

                float u = phi_diff_index(phi_diff)   / 179.0;
                float v = theta_diff_index(theta_diff) / 89.0;
                float w = theta_half_index(theta_half) / 89.0;

                vec3 brdfValue = texture(brdfData, vec3(u, v, w)).rgb;

                vec3 color = brdfValue * NdotL * 100.0;
                color = color / (color + vec3(1.0));
                gl_FragColor = vec4(color, 1.0);
            }
        `;

        super({
            uniforms: {
                brdfData:      { value: brdfTexture },
                lightPosition: { value: new THREE.Vector3(5, 5, 5) },
                viewPosition:  { value: new THREE.Vector3(0, 0, 5) },
            },
            vertexShader,
            fragmentShader
        });
    }
}

// ── Renderer wrapper exposed to Julia/Bonito ─────────────────────────────────
export function createMerlRenderer(container) {
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

    // Visible "light bulb" marker
    const lightOrbitRadius = 3.0;
    const lightMesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.06, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0xffffaa })
    );
    scene.add(lightMesh);

    // Sphere — placeholder material until BRDF data arrives
    const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(1, 64, 64),
        new THREE.MeshStandardMaterial({ color: 0x808080 })
    );
    scene.add(sphere);
    scene.add(new THREE.AmbientLight(0xffffff, 0.4));

    function setLightDir(theta, phi) {
        const x = Math.sin(theta) * Math.cos(phi) * lightOrbitRadius;
        const y = Math.sin(theta) * Math.sin(phi) * lightOrbitRadius;
        const z = Math.cos(theta)                 * lightOrbitRadius;
        lightMesh.position.set(x, y, z);
    }
    setLightDir(Math.PI / 4, 0);

    async function loadBufferBase64(b64) {
        if (!b64) return;
        try {
            const bin = atob(b64);
            const bytes = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
            const tex = await MERLBRDFMaterial.loadBRDFFromBuffer(bytes.buffer);
            const mat = new MERLBRDFMaterial(tex);
            sphere.material.dispose();
            sphere.material = mat;
        } catch (e) {
            console.error('[merl_renderer] failed to load BRDF buffer:', e);
        }
    }

    let raf = 0;
    function animate() {
        raf = requestAnimationFrame(animate);
        if (sphere.material.uniforms && sphere.material.uniforms.lightPosition) {
            sphere.material.uniforms.lightPosition.value.copy(lightMesh.position);
            sphere.material.uniforms.viewPosition.value.copy(camera.position);
        }
        controls.update();
        renderer.render(scene, camera);
    }
    animate();

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
        sphere.material.dispose();
        sphere.geometry.dispose();
        renderer.dispose();
        if (renderer.domElement.parentNode === container) {
            container.removeChild(renderer.domElement);
        }
    }

    return { setLightDir, loadBufferBase64, resize, dispose };
}
