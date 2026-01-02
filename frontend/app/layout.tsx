export const metadata = {
  title: "SOTAforge",
  description: "Generate State-of-the-Art summaries for any research topic",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
