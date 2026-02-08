import Image from "next/image";
import Link from "next/link";

export default function Logo({
  href = "/",
  className = "",
  priority = false,
}) {
  return (
   <Link
  href={href}
  className={`relative block 
    w-[180px] lg:w-[260px] 
    h-[80px] lg:h-[110px] 
    ${className}`}
   >
      <Image
        src="/images/main-logo.png"
        alt="Worksbot Logo"
        fill
        priority={priority}
        draggable={false}
        className="object-contain"
      />
    </Link>
  );
}
