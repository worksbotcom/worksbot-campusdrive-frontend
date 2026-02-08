"use client"; // Enables client-side rendering
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import useStore from "@/store/store";
import { getToggleState, updateToggleState } from "@/utils/api";
import Link from "next/link";
import { useRouter } from "next/navigation";
import React, { useEffect, useState } from "react";
import { Federo } from 'next/font/google';
import Logo from "@/components/common/logo";

const federo = Federo({
  weight: '400',
  subsets: ['latin'],
  display: 'swap',
});

const Header = () => {
  const { logout } = useStore();
  const router = useRouter();
  const [isActive, setIsActive] = useState(false); // Toggle state
  const [loading, setLoading] = useState(true); // Track loading state

  // Fetch the toggle state on mount
  useEffect(() => {
    const fetchToggle = async () => {
      try {
        const token = sessionStorage.getItem("token"); // Retrieve token
        const response = await getToggleState(token);
        setIsActive(response.data.isActive); // Adjust based on actual API response
        // Set initial state
      } catch (error) {
        console.error("Error fetching toggle state:", error);
      } finally {
        setLoading(false); // Stop loading
      }
    };

    fetchToggle();
  }, []);

  // Handle switch toggle
  const handleToggle = async (checked) => {
    setIsActive(checked); // Optimistically update state immediately

    try {
      const token = sessionStorage.getItem("token"); // Retrieve token
      await updateToggleState(token, checked); // Update backend state
    } catch (error) {
      console.error("Error updating toggle state:", error);
      setIsActive(!checked); // Revert state if error occurs
    }
  };

  // Logout handler
  const handleLogout = () => {
    logout();
    router.push("/sign-in");
  };

  return (
    <header className="flex justify-between items-center p-4">
      <div className="flex  items-center gap-14">
         <Link
                            className={`panel text-none text-4xl mb-8 text-red-500 ${federo.className}`}
                            href={`/`}
                            style={{ width: 200 }}
                          >
                           <Logo/>
                          </Link>
        <Link href={"/admin"}>All User</Link>
        <Link href={"/admin/search"}>Search User</Link>
      </div>
      <div className="flex items-center gap-4">
        {/* Ping indicator, changes color based on `isActive` */}
        <span className="relative flex h-3 w-3">
          <span
            className={`animate-ping absolute inline-flex h-full w-full rounded-full 
              ${isActive ? "bg-green-400" : "bg-red-400"} opacity-75`}
          ></span>
          <span
            className={`relative inline-flex rounded-full h-3 w-3 
              ${isActive ? "bg-green-500" : "bg-red-500"}`}
          ></span>
        </span>

        {/* Disable switch while loading */}
        <Switch
          checked={isActive}
          onCheckedChange={handleToggle}
          disabled={loading} // Prevent interaction during loading
        />

        <Button
          variant="outline"
          className="border-primary text-primary"
          onClick={handleLogout}
        >
          Logout
        </Button>
      </div>
    </header>
  );
};

export default Header;
