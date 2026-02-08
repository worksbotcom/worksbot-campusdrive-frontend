import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import "../public/css/main.css";
export const metadata = {
  title:
    "WORKSBOT",
  description:
    "WORKSBOT Assessment",
    keywords:"WORKSBOT Assessment"
};
export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body
        className={` antialiased`}
      >
       <main>
       {children}
        </main> 
        <Toaster/>
      </body>
    </html>
  );
}
