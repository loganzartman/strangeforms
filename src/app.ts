import tgpu from "typegpu";
import {
  struct,
  vec2f,
  vec3f,
  vec4f,
  mat4x4f,
  f32,
  arrayOf,
} from "typegpu/data";
import { vec2, vec3, mat4, utils } from "wgpu-matrix";
import { wgsl } from "./wgsl";
import { Timing } from "./Timing";
import { clamp } from "./clamp";

const Blob = struct({
  position: vec3f,
  radius: f32,
  color: vec4f,
}).$name("Blob");

const Uniforms = struct({
  time: f32,
  cameraPos: vec3f,
  lookPos: vec3f,
  upVec: vec3f,
  cameraMat: mat4x4f,
  invCameraMat: mat4x4f,
  projMat: mat4x4f,
  invProjMat: mat4x4f,
}).$name("Uniforms");

export async function init({ container }: { container: HTMLDivElement }) {
  const timing = new Timing();
  const root = await tgpu.init();
  const canvas = container.querySelector<HTMLCanvasElement>("#app-canvas")!;
  const context = canvas.getContext("webgpu");

  if (!context) {
    throw new Error("Failed to get webgpu context");
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: root.device,
    format,
    alphaMode: "premultiplied",
  });

  const uniformsBuffer = root.createBuffer(Uniforms).$usage("uniform");

  const nBlobs = 11;
  const blobBuffer = root.createBuffer(arrayOf(Blob, nBlobs)).$usage("storage");
  const randomizeBlobs = () => {
    const blobs = Array.from({ length: nBlobs }, () => {
      const position = vec3f(
        Math.random() * 2 - 1,
        Math.random() * 2 - 1,
        Math.random() * 2 - 1,
      );
      return {
        position,
        radius: Math.random() * 0.3 + 0.1,
        color: vec3.normalize(
          vec3.add(vec3.random(0.5, vec3f()), vec3f(0.5), vec3f()),
          vec4f(0, 0, 0, 1),
        ),
      };
    });
    blobBuffer.write(blobs);
  };
  randomizeBlobs();

  const renderModule = root.device.createShaderModule({
    code: wgsl`
      @group(0) @binding(0) var<uniform> uniforms: Uniforms;
      @group(0) @binding(1) var<storage, read> blobs: array<Blob>;

      struct Uniforms {
        time: f32,
        cameraPos: vec3<f32>,
        lookPos: vec3<f32>,
        upVec: vec3<f32>,
        cameraMat: mat4x4<f32>,
        invCameraMat: mat4x4<f32>,
        projMat: mat4x4<f32>,
        invProjMat: mat4x4<f32>,
      };

      struct Blob {
        position: vec3<f32>,
        radius: f32,
        color: vec4<f32>,
      };

      struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) color: vec4<f32>,
      };

      @vertex fn vs(
        @builtin(vertex_index) vertexIndex : u32,
      ) -> VertexOutput {
        let blob = blobs[vertexIndex];
        let pos = uniforms.projMat * uniforms.cameraMat * vec4f(blob.position, 1.0);
        return VertexOutput(pos, blob.color);
      }

      @fragment fn fs(
        input: VertexOutput,
      ) -> @location(0) vec4<f32> {
        return vec4f(input.color.rgb * input.color.a, input.color.a);
      }
    `,
  });

  const raymarchModule = root.device.createShaderModule({
    code: wgsl`
      @group(0) @binding(0) var<uniform> uniforms: Uniforms;
      @group(0) @binding(1) var<storage, read> blobs: array<Blob>;

      struct Uniforms {
        time: f32,
        cameraPos: vec3<f32>,
        lookPos: vec3<f32>,
        upVec: vec3<f32>,
        cameraMat: mat4x4<f32>,
        invCameraMat: mat4x4<f32>,
        projMat: mat4x4<f32>,
        invProjMat: mat4x4<f32>,
      };

      struct Blob {
        position: vec3<f32>,
        radius: f32,
        color: vec4<f32>,
      };

      struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) clipPos: vec2f,
      };

      struct SmoothMin {
        min: f32,
        blend: f32,
      };

      struct SDFValue {
        dist: f32,
        color: vec4f,
      };

      @vertex fn vs(
        @builtin(vertex_index) vertexIndex : u32,
      ) -> VertexOutput {
        // full-screen quad
        let pos = array(
          vec2f(-1, -1),
          vec2f(1, -1),
          vec2f(-1, 1),
          vec2f(-1, 1),
          vec2f(1, -1),
          vec2f(1, 1)
        );

        let uv = array(
          vec2f(0, 1),
          vec2f(1, 1),
          vec2f(0, 0),
          vec2f(0, 0),
          vec2f(1, 1),
          vec2f(1, 0)
        );

        var output: VertexOutput;
        output.position = vec4f(pos[vertexIndex], 0.0, 1.0);
        output.clipPos = pos[vertexIndex];
        return output;
      }

      fn sdBlob(p: vec3f, blob: Blob) -> f32 {
        let t = uniforms.time;
        let x = blob.position;
        let scale = 0.05;
        let offset = vec3f(
          sin(x.x * 3.52 + t * 1.01) * cos(x.z * 4.78 + t * 2.12),
          sin(x.y * 5.31 + t * 0.83) * cos(x.x * 3.98 + t * 3.12),
          sin(x.z * 4.31 + t * 3.17) * cos(x.y * 5.11 + t * 1.51),
        ) * scale;
        return length(p - blob.position + offset) - blob.radius;
      }

      // quadratic polynomial
      fn smin(a: f32, b: f32, k: f32) -> SmoothMin {
        let h = 1.0 - min(abs(a - b) / (4.0 * k), 1.0);
        let w = h * h;
        let m = w * 0.5;
        let s = w * k;
        if (a < b) {
          return SmoothMin(a - s, m);
        } else {
          return SmoothMin(b - s, 1.0 - m);
        }
      }

      fn sdf(p: vec3f) -> SDFValue {
        var result = SDFValue();
        result.dist = 999999.0;
        result.color = blobs[0].color;

        for (var i = 0u; i < arrayLength(&blobs); i += 1) {
          let blob = blobs[i];
          let m = smin(result.dist, sdBlob(p, blob), 0.2);
          result.dist = m.min;
          result.color = mix(result.color, blob.color, m.blend);
        }

        return result;
      }

      fn sdfNormal(p: vec3f) -> vec3f {
        let h = 0.0001;
        let k = vec2f(1, -1);
        return normalize(
          k.xyy * sdf(p + k.xyy * h).dist +
          k.yxy * sdf(p + k.yxy * h).dist +
          k.yyx * sdf(p + k.yyx * h).dist +
          k.xxx * sdf(p + k.xxx * h).dist
        );
      }

      @fragment fn fs(
        input: VertexOutput,
      ) -> @location(0) vec4f {
        let fragClipPos = input.clipPos.xy;
        let fragCameraPos =
          (uniforms.invProjMat * vec4f(fragClipPos, 0.0, 1.0)).xyz;
        let fragWorldPos =
          (uniforms.invCameraMat * vec4f(fragCameraPos, 1.0)).xyz;
        let rayDir = normalize(fragWorldPos - uniforms.cameraPos);

        let epsilon = 0.001;
        var p = vec3f();
        var t = 0.0;
        var result = SDFValue();
        var i = 0;
        for (; i < 256; i += 1) {
          p = uniforms.cameraPos + rayDir * t;
          result = sdf(p);
          if (result.dist < epsilon) {
            break;
          }
          t += result.dist;

          if (t > 100.0) {
            break;
          }
        }

        if (result.dist > epsilon) {
          return vec4f(normalize(abs(rayDir.xyz) * 0.25 + 0.75), 1.0);
        }
        return result.color;
      }
    `,
  });

  const renderBindGroupLayout = tgpu.bindGroupLayout({
    uniforms: { uniform: uniformsBuffer.dataType },
    blobs: { storage: blobBuffer.dataType },
  });

  const renderBindGroup = renderBindGroupLayout.populate({
    uniforms: uniformsBuffer,
    blobs: blobBuffer,
  });

  const raymarchBindGroupLayout = tgpu.bindGroupLayout({
    uniforms: { uniform: uniformsBuffer.dataType },
    blobs: { storage: blobBuffer.dataType },
  });

  const raymarchBindGroup = raymarchBindGroupLayout.populate({
    uniforms: uniformsBuffer,
    blobs: blobBuffer,
  });

  const renderPipeline = root.device.createRenderPipeline({
    layout: root.device.createPipelineLayout({
      bindGroupLayouts: [root.unwrap(renderBindGroupLayout)],
    }),
    vertex: {
      module: renderModule,
    },
    fragment: {
      module: renderModule,
      targets: [
        {
          format,
          blend: {
            color: {
              srcFactor: "one",
              dstFactor: "one",
            },
            alpha: {
              srcFactor: "one",
              dstFactor: "one-minus-src-alpha",
            },
          },
        },
      ],
    },
    primitive: {
      topology: "point-list",
    },
  });

  const raymarchPipeline = root.device.createRenderPipeline({
    layout: root.device.createPipelineLayout({
      bindGroupLayouts: [root.unwrap(raymarchBindGroupLayout)],
    }),
    vertex: {
      module: raymarchModule,
    },
    fragment: {
      module: raymarchModule,
      targets: [{ format }],
    },
  });

  const renderPassDescriptor = {
    label: "Render pass descriptor",
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        clearValue: [0.0, 0.0, 0.0, 1.0],
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  } satisfies GPURenderPassDescriptor;

  let aspectRatio = 1;
  const handleResize = () => {
    canvas.width = Math.ceil(window.innerWidth * window.devicePixelRatio * timing.renderScale);
    canvas.height = Math.ceil(window.innerHeight * window.devicePixelRatio * timing.renderScale);
    aspectRatio = canvas.width / canvas.height;
  };
  window.addEventListener("resize", handleResize);
  handleResize();

  const origin = vec3f(0, 0, 0);
  const pointer = {
    down: false,
    pos: vec2f(0, 0),
    prevPos: vec2f(0, 0),
  };
  const orbitCam = {
    dist: 8,
    ax: 0,
    az: 0,
    vdist: 0,
    vax: -2,
    vaz: 20,
    pos: vec3f(0, 0, 0),
  };
  const updatePointerAndCamera = (dt: number) => {
    const pointerVel = vec2.mulScalar(
      vec2.divScalar(vec2.sub(pointer.pos, pointer.prevPos), dt),
      -50,
    );
    vec2.copy(pointer.pos, pointer.prevPos);

    // decelerate
    orbitCam.vax *= Math.pow(0.015, dt);
    orbitCam.vaz *= Math.pow(0.015, dt);
    orbitCam.vdist *= Math.pow(0.015, dt);

    // orbitCam.vaz += 2 * dt;

    if (pointer.down) {
      orbitCam.vax += pointerVel[1] * dt;
      orbitCam.vaz += pointerVel[0] * dt;
    }

    orbitCam.ax = clamp(
      orbitCam.ax + orbitCam.vax * dt,
      -Math.PI / 2 + 0.0001,
      Math.PI / 2 - 0.0001,
    );
    orbitCam.az += orbitCam.vaz * dt;
    orbitCam.dist += orbitCam.vdist * dt;

    vec3.set(0, -orbitCam.dist, 0, orbitCam.pos);
    vec3.rotateX(orbitCam.pos, origin, orbitCam.ax, orbitCam.pos);
    vec3.rotateZ(orbitCam.pos, origin, orbitCam.az, orbitCam.pos);
  };
  window.addEventListener(
    "pointerdown",
    (e) => {
      pointer.pos[0] = e.clientX / window.innerWidth;
      pointer.pos[1] = e.clientY / window.innerHeight;
      pointer.down = true;
    },
    false,
  );
  window.addEventListener(
    "pointermove",
    (e) => {
      pointer.pos[0] = e.clientX / window.innerWidth;
      pointer.pos[1] = e.clientY / window.innerHeight;
    },
    false,
  );
  window.addEventListener(
    "pointerup",
    () => {
      pointer.down = false;
    },
    false,
  );
  window.addEventListener(
    "pointercancel",
    () => {
      pointer.down = false;
    },
    false,
  );

  const makeUniforms = () => {
    const time = performance.now() / 1000;
    const cameraPos = orbitCam.pos;
    const lookPos = vec3f(0, 0, 0);
    const upVec = vec3f(0, 0, 1);
    const cameraMat = mat4.lookAt(cameraPos, lookPos, upVec, mat4x4f());
    const projMat = mat4.perspective(
      utils.degToRad(40),
      aspectRatio,
      0.1,
      1000,
      mat4x4f(),
    );
    const invCameraMat = mat4.inverse(cameraMat, mat4x4f());
    const invProjMat = mat4.inverse(projMat, mat4x4f());
    return {
      time,
      cameraPos,
      lookPos,
      upVec,
      cameraMat,
      invCameraMat,
      projMat,
      invProjMat,
    };
  };

  const scheduleFrame = () => {
    requestAnimationFrame(() => handleFrame().catch(e => {
      console.error(e);
    }))
  }

  const handleFrame = async () => {
    timing.startFrame();
    const dt = timing.getLastFrameTime();
    if (timing.updateRenderScale()) {
      console.log(`New render scale: ${timing.renderScale}`);
      handleResize();
    }

    updatePointerAndCamera(dt);

    uniformsBuffer.write(makeUniforms());

    renderPassDescriptor.colorAttachments[0].view = context
      .getCurrentTexture()
      .createView();

    const encoder = root.device.createCommandEncoder();
    const renderPass = encoder.beginRenderPass(renderPassDescriptor);

    // points
    // renderPass.setPipeline(renderPipeline);
    // renderPass.setBindGroup(0, root.unwrap(renderBindGroup));
    // renderPass.draw(nBlobs);

    // raymarched
    renderPass.setPipeline(raymarchPipeline);
    renderPass.setBindGroup(0, root.unwrap(raymarchBindGroup));
    renderPass.draw(6);

    renderPass.end();

    root.device.queue.submit([encoder.finish()]);
    await root.device.queue.onSubmittedWorkDone();

    scheduleFrame();
  };
  scheduleFrame();
}