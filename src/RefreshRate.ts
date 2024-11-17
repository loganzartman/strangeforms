import { median } from "./median";

export class RefreshRate {
  running = false;
  lastTime: number | null = null;
  refreshRate: number | null = null;
  frameTimes: number[] = [];
  bufferSize = 10;

  now() {
    return performance.now() / 1000;
  }

  frame() {
    const now = this.now();
    if (this.lastTime !== null) {
      const dt = now - this.lastTime;
      const instantRate = 1 / dt;
      
      this.frameTimes.push(instantRate);
      while (this.frameTimes.length > this.bufferSize) {
        this.frameTimes.shift();
      }

      if (this.frameTimes.length === this.bufferSize) {
        this.refreshRate = median(this.frameTimes);
      }
    }
    this.lastTime = now;

    if (this.running) {
      requestAnimationFrame(() => this.frame());
    }
  }

  start() {
    this.running = true;
    this.lastTime = null;
    this.refreshRate = null;
    this.frame();
  }

  stop() {
    this.running = false;
  }

  get() {
    return this.refreshRate;
  }
}