/** @type {import('next').NextConfig} */
const nextConfig = {
    typescript: {
        ignoreBuildErrors: true, // 🔥 빌드 오류 무시
      },
      eslint: {
        ignoreDuringBuilds: true, // 🔥 ESLint 오류 무시
      },
      reactStrictMode: false, // 🔥 개발 모드처럼 React Strict Mode 끄기 (필요한 경우)
};

export default nextConfig;
