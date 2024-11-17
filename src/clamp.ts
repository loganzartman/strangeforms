export function clamp(x: number, a: number, b: number): number {
  return Math.max(a, Math.min(x, b));
}
