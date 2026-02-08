import SignUpCard from "@/components/common/SignUpCard";
import { Federo } from 'next/font/google';
import Link from 'next/link';
import Logo from "@/components/common/logo";

const federo = Federo({
  weight: '400',
  subsets: ['latin'],
  display: 'swap',
});

export default function page() {
  return (
    <div className="w-full lg:grid lg:min-h-[600px] lg:grid-cols-2 xl:min-h-screen">
      {/* Left side with background image and overlay */}
      <div
        className="hidden lg:block relative bg-cover bg-center"
        style={{ backgroundImage: "url('/interview1.jpg')" }}
      >
        {/* Black gradient overlay from the top */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />

        {/* Logo positioned on top */}
        <div className="relative flex items-center justify-center p-8">
          <Link
            className={`text-4xl text-red-500 ${federo.className}`}
            href={`/`}
          >
            <Logo className=""/>
          </Link>
        </div>
      </div>

      {/* Right side with the sign-up card */}
      <div className="flex items-center max-h-screen justify-center py-12 overflow-y-auto">
        <SignUpCard />
      </div>
    </div>
  );
}