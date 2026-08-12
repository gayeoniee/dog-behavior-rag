/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // 휴대폰 등 LAN의 다른 기기에서 개발 서버를 열 때 필요하다.
  // 지금은 경고지만 다음 메이저 버전부터는 이게 없으면 /_next/* 요청이 막힌다.
  // IP가 DHCP로 바뀌므로 사설 대역을 통째로 넣는다 (개발 서버 한정).
  allowedDevOrigins: ["192.168.0.0/16", "10.0.0.0/8", "172.16.0.0/12"],
};

export default nextConfig;
