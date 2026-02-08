import LoginCard from '@/components/common/LoginCard'
import React from 'react'
import { Federo } from 'next/font/google';
import Link from 'next/link';
import Logo from '@/components/common/logo';

const federo = Federo({
  weight: '400',
  subsets: ['latin'],
  display: 'swap',
});

export default function Page() {
  return (
    <div className="w-full lg:grid lg:min-h-[600px] lg:grid-cols-2 xl:min-h-screen">
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
            <Logo/>
          </Link>
        </div>
      </div>

      {/* Right Section */}
      <div className="flex items-center justify-center py-12">
        <LoginCard />
      </div>
    </div>
  );
}
