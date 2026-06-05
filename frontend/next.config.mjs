/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Hide the Next.js dev indicator badge (bottom-left logo).
  devIndicators: false,
  // Standalone output (for the Docker image) only when explicitly requested —
  // it uses symlinks that fail on Windows and is unnecessary on Vercel.
  output: process.env.BUILD_STANDALONE ? "standalone" : undefined,
};

export default nextConfig;
