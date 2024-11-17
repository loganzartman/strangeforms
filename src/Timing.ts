import { clamp } from "./clamp";
import { median } from "./median";

export class Timing {
  lastFrameStart = this.now();
  frameTimes: number[] = [];
  bufferSize = 15;
  targetFps = 60;

  renderScale = 0.5;
  minRenderScale = 0.1;
  maxRenderScale = 1;

  frames = 0;
  interval = 30;

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
    return median(this.frameTimes);
  }

  getFramerate(): number | null {
    const frameTime = this.getFrameTime();
    if (frameTime === null) {
      return null;
    }
    return 1 / frameTime;
  }

  updateRenderScale(): boolean {
    const {targetFps} = this;
    if (!targetFps) {
      return false;
    }
    if (this.frames < this.interval) {
      return false;
    }
    const framerate = this.getFramerate();
    if (framerate === null) {
      return false;
    }
    this.frames = 0;
    
    console.log(framerate);

    const fpsLag = targetFps - framerate;
    if (fpsLag > 10  && this.renderScale > this.minRenderScale) {
      this.renderScale = clamp(this.renderScale - 0.05, this.minRenderScale, this.maxRenderScale);
      return true;
    }
    if (fpsLag < 1 && this.renderScale < this.maxRenderScale) {
      this.renderScale = clamp(this.renderScale + 0.05, this.minRenderScale, this.maxRenderScale);
      return true;
    }
    return false;
  }
}