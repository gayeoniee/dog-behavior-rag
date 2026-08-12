import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GY_RAG — 반려동물 훈련 상담",
  description: "기관·학술 자료 기반 반려동물 훈련/문제행동 상담",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
