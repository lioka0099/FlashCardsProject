// Shared topic accent palette so a topic's color is the same in the progress
// panel and the history list (previously each cycled colors by array index,
// which drifted since the two views order/dedupe topics differently).
const TOPIC_COLORS = ["#6f42c1", "#2db7c5", "#22c55e", "#f59e0b", "#ef4444", "#a78bfa", "#0ea5e9", "#ec4899"];

export function getTopicColor(label: string): string {
  let hash = 0;
  for (let i = 0; i < label.length; i++) {
    hash = (hash * 31 + label.charCodeAt(i)) | 0;
  }
  return TOPIC_COLORS[Math.abs(hash) % TOPIC_COLORS.length];
}
