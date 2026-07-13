import { getToken } from "@/lib/session/token";

export async function fetchSourceBlob(url: string): Promise<Blob> {
  const headers = new Headers();
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const response = await fetch(url, { headers });
  if (!response.ok) {
    throw new Error(`Failed to fetch source document (${response.status})`);
  }
  return response.blob();
}
