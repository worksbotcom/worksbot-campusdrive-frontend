import Logo from "@/components/common/logo";
import VerifyOtpCard from "@/components/common/VerifyOtpCard"
import { Federo } from 'next/font/google';
import Link from 'next/link';

const federo = Federo({
  weight: '400',
  subsets: ['latin'],
  display: 'swap',
});

export default function Page() {
  return (
    <div className="w-full lg:grid lg:min-h-[600px] lg:grid-cols-2 xl:min-h-screen">
    <div className="hidden lg:flex lg:flex-col bg-primary-foreground p-4">
    
      <div className="flex flex-col max-w-lg mx-auto justify-center flex-1 ">
 <Link
                    className={`panel text-none text-4xl mb-8 text-primary ${federo.className}`}
                    href={`/`}
                    style={{ width: 200 }}
                  >
                   <Logo/>
                  </Link>        
      </div>
    </div>

      <div className="flex h-screen items-center justify-center py-12">
       <VerifyOtpCard/>
      </div>
      </div>
    
  )
}
