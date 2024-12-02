import { average } from "./util";

export class Timing {
  lastFrameStart = this.now();
  lastRenderStart = this.now();
  frameTimes: number[] = [];
  renderTimes: number[] = [];
  bufferSize = 15;
  targetFps = 60;

  renderScale = 0.5;
  minRenderScale = 0.1;
  maxRenderScale = 1;

  frames = 0;

  gpuTimestamp: {
    querySet: GPUQuerySet;
    resolveBuffer: GPUBuffer;
    resultBuffer: GPUBuffer;
  } | null = null;

  constructor({ device }: { device: GPUDevice }) {
    if (device.features.has("timestamp-query")) {
      console.log("GPU device has timestamp-query enabled");
      const querySet = device.createQuerySet({
        type: "timestamp",
        count: 2,
      });
      const resolveBuffer = device.createBuffer({
        size: querySet.count * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      const resultBuffer = device.createBuffer({
        size: resolveBuffer.size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      this.gpuTimestamp = { querySet, resolveBuffer, resultBuffer };
    }
  }

  getTimestampWrites() {
    if (!this.gpuTimestamp) {
      return undefined;
    }
    return {
      querySet: this.gpuTimestamp.querySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    };
  }

  encodeGetResult(encoder: GPUCommandEncoder) {
    if (!this.gpuTimestamp) {
      return;
    }
    const { querySet, resolveBuffer, resultBuffer } = this.gpuTimestamp;
    encoder.resolveQuerySet(querySet, 0, querySet.count, resolveBuffer, 0);
    encoder.copyBufferToBuffer(
      resolveBuffer,
      0,
      resultBuffer,
      0,
      resultBuffer.size
    );
  }

  async resolveResult() {
    if (!this.gpuTimestamp) {
      return;
    }
    const { resultBuffer } = this.gpuTimestamp;
    await resultBuffer.mapAsync(GPUMapMode.READ);
    const times = new BigUint64Array(resultBuffer.getMappedRange());
    // from the spec: https://gpuweb.github.io/gpuweb/#timestamp
    // "The physical device may reset the timestamp counter occasionally,
    // which can result in unexpected values such as negative deltas between
    // timestamps that logically should be monotonically increasing. These
    // instances should be rare and can safely be ignored."
    //
    // TODO: on apple silicon, these values seem like complete garbage.
    // the difference isn't positive even on average.
    const duration = Number(times[1] - times[0]) / 1e9;
    this.renderTimes.push(duration);
    resultBuffer.unmap();
  }

  now() {
    return performance.now() / 1000;
  }

  startFrame() {
    ++this.frames;

    const now = this.now();
    const lastFrameTime = now - this.lastFrameStart;
    this.lastFrameStart = now;

    this.frameTimes.push(lastFrameTime);
    while (this.frameTimes.length > this.bufferSize) {
      this.frameTimes.shift();
    }
  }

  getLastFrameTime(): number {
    if (this.frameTimes.length === 0) {
      return 0;
    }
    return this.frameTimes.at(-1)!;
  }

  getFrameTime(): number | null {
    if (this.frameTimes.length < this.bufferSize) {
      return null;
    }
    return average(this.frameTimes);
  }

  getRenderTime(): number | null {
    if (this.renderTimes.length < this.bufferSize) {
      return null;
    }
    return average(this.renderTimes);
  }

  getFramerate(): number | null {
    const frameTime = this.getFrameTime();
    if (frameTime === null) {
      return null;
    }
    return 1 / frameTime;
  }

  updateRenderScale(): boolean {
    const { targetFps } = this;
    if (!targetFps) {
      return false;
    }

    const frameTime = this.getFrameTime();
    const renderTime = this.getRenderTime();
    if (frameTime === null || renderTime === null) {
      return false;
    }

    console.log(this.frameTimes.length, this.renderTimes.length);

    this.frameTimes.length = 0;
    this.renderTimes.length = 0;

    console.log({ frameTime, renderTime });
    return false;
  }
}
