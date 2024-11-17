import { clamp } from "./clamp";
import { RefreshRate } from "./RefreshRate";

export class Timing {
  refreshRate = new RefreshRate();
  lastFrameStart = this.now();
  frameTime: number | null = null;
  lastFrameTime: number | null = null;
  smoothing = 0.5;

  renderScale = 0.5;
  minRenderScale = 0.1;
  maxRenderScale = 1;

  frames = 0;
  interval = 20;

  constructor() {
    this.refreshRate.start();
  }

  now() {
    return performance.now() / 1000;
  }

  startFrame() {
    ++this.frames;

    const now = this.now();
    this.lastFrameTime = now - this.lastFrameStart;
    this.lastFrameStart = now;

    if (this.frameTime) {
      this.frameTime = (this.frameTime * this.smoothing) + (this.lastFrameTime * (1 - this.smoothing));
    } else {
      this.frameTime = this.lastFrameTime;
    }
  }

  getLastFrameTime() {
    if (this.lastFrameTime === null) {
      return 0;
    }
    return this.lastFrameTime;
  }

  getFramerate() {
    if (this.frameTime === null) {
      throw new Error('No frameTime available');
    }
    return 1 / this.frameTime;
  }

  updateRenderScale(): boolean {
    const targetFps = this.refreshRate.get();

    if (!targetFps) {
      return false;
    }
    if (this.frames < this.interval) {
      return false;
    }
    if (this.frameTime === null) {
      return false;
    }
    this.frames = 0;

    const fpsLag = targetFps - this.getFramerate();
    if (fpsLag > targetFps * 0.2 && this.renderScale > this.minRenderScale) {
      this.renderScale = clamp(this.renderScale * 0.8, this.minRenderScale, this.maxRenderScale);
      return true;
    }
    if (fpsLag < 5 && this.renderScale < this.maxRenderScale) {
      this.renderScale = clamp(this.renderScale * 1.2, this.minRenderScale, this.maxRenderScale);
      return true;
    }
    return false;
  }
}