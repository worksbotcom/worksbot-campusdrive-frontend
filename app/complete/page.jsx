"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import useStore from "@/store/store";
import { Loader2 } from "lucide-react";

import React from 'react'

const page = () => {
  const { logout, clearSession } = useStore(); // Use the logout function from your store
  const [countdown, setCountdown] = useState(5); // Initial countdown value (5 seconds)
  const router = useRouter();

  useEffect(() => {
    // Start the countdown timer
    const timer = setInterval(() => {
      setCountdown((prev) => prev - 1); // Decrement countdown every second
    }, 1000);

    // After 5 seconds, log the user out and clear session
    if (countdown === 0) {
      clearInterval(timer); // Stop the countdown
      logout(); // Clear the session and logout the user
      clearSession();
      router.push("/sign-up"); // Redirect to sign-in page
    }

    // Cleanup interval on component unmount
    return () => clearInterval(timer);
  }, [countdown, logout, router]);


  return (
    <div className="h-screen flex flex-col items-center justify-center text-center">
      <h1 className="text-2xl font-bold mb-4">Thank you for attending the assessment!</h1>
      <p>Logging you out in {countdown} seconds...</p>
            <Loader2 className="animate-spin w-6 h-6 mt-4" />
    </div>
  )
}

export default page