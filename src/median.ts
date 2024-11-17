export function median(a: number[]): number {
  if (!a.length) throw new Error('Empty array');
  return [...a].sort()[Math.floor(a.length / 2)];
}