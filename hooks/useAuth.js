"use client";
import { useEffect, useState } from "react";
import useStore from "@/store/store";
import { useRouter } from "next/navigation";

const useAuth = () => {
  const { user, setToken, setUser } = useStore(); // Assuming you have a setUser function
  const [token, setLocalToken] = useState(null);

  const router = useRouter();

  useEffect(() => {
    if (typeof window !== "undefined") {
      const storedToken = sessionStorage.getItem("token");
      if (storedToken) {
        setLocalToken(storedToken);
        setToken(storedToken); // Set token in global state/store immediately
      }
    }
  }, [setToken]);

  useEffect(() => {
    if (!token) {
      // If there's no token, redirect to sign-in
      router.push("/sign-in");
    } else if (!user) {
      // If there's a token but no user, wait for user to be set before redirecting
      setToken(token); // Set token again just in case
    }
  }, [token, user, setToken, router]);

  return { token, user };
};

export default useAuth;
