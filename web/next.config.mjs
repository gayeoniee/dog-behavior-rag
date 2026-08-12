/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // 휴대폰 등 LAN의 다른 기기에서 개발 서버를 열 때 필요하다. 없으면 /_next/*가
  // 차단돼 페이지 뼈대만 오고 JS·CSS가 안 내려온다 (Next 15에서 실제로 막혔다).
  //
  // **CIDR로 쓰면 안 된다.** Next는 이 값을 네트워크 대역이 아니라 호스트명 문자열로
  // 매칭해서 "192.168.0.0/16"은 호스트 "192.168.0.244"와 영영 일치하지 않는다.
  // IP가 DHCP로 바뀌므로 와일드카드 패턴으로 사설 대역을 덮는다 (개발 서버 한정).
  allowedDevOrigins: ["192.168.*.*", "10.*.*.*", "172.16.*.*", "172.17.*.*"],
};

export default nextConfig;
