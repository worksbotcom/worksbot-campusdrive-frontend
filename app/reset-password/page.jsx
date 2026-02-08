import ResetPasswordCard from '@/components/common/ResetPasswordCard'
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
        style={{ backgroundImage: "url('/interview.jpg')" }}
      >
        {/* Black gradient overlay from the top */}
        <div className="absolute inset-0 bg-gradient-to-b from-black/60 to-transparent" />

        {/* Logo positioned on top */}
        <div className="relative p-8">
          <Link
            className={`text-4xl text-primary ${federo.className}`}
            href={`/`}
          >
            <Logo/>
          </Link>
        </div>
      </div>

    <div className="flex h-screen items-center justify-center py-12">
     <ResetPasswordCard/>
    </div>
    </div>
  
  )
}

