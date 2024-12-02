export function average(a: number[]): number {
  if (!a.length) throw new Error('Empty array');
  const invLen = 1 / a.length;
  let avg = 0;
  for (const x of a) {
    avg += x * invLen;
  }
  return avg;
}

export function clamp(x: number, a: number, b: number): number {
  return Math.max(a, Math.min(x, b));
}

export function median(a: number[]): number {
  if (!a.length) throw new Error('Empty array');
  return [...a].sort()[Math.floor(a.length / 2)];
}