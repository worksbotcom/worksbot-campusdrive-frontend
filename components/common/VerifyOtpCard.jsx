"use client";
import { useEffect, useState } from "react";
import { InputOTP, InputOTPGroup, InputOTPSlot } from "@/components/ui/input-otp";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";
import Link from "next/link";
import { verifyOTP } from "@/utils/api";
import useStore from "@/store/store";
import { useRouter } from "next/navigation";

const VerifyOtpCard = () => {
  const [loading, setLoading] = useState(false);
  const [value, setValue] = useState("");
  const { email, setToken} = useStore();
  const router = useRouter();

  if (!email) {
    router.push("/sign-in");
  }

  const handleOTPVerification = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const verifyOutput = await verifyOTP(email, value);
      if (verifyOutput?.status === "success") {
        setToken(verifyOutput?.data?.token);
        router.push("/");
        setLoading(false);
      } else {
        setLoading(false);
      }
    } catch (error) {
      console.error("Error verifying OTP", error);
    }
    setLoading(false);
  };

  return (
    <div className="mx-auto grid w-[350px] gap-6">
      <div className="grid gap-2 text-center">
        <h1 className="text-3xl font-bold">Verify Otp</h1>
        <p className="text-muted-foreground">
          Enter your OTP to verify your account
        </p>
      </div>
      <form onSubmit={handleOTPVerification} className="grid gap-4">
        <div className="grid gap-2 justify-center">
          <InputOTP
            maxLength={6}
            value={value}
            onChange={(value) => setValue(value)}
          >
            <InputOTPGroup>
              {[...Array(6)].map((_, idx) => (
                <InputOTPSlot key={idx} index={idx} />
              ))}
            </InputOTPGroup>
          </InputOTP>
        </div>
        <Button type="submit" className="w-full">
          {loading ? <Loader2 className="animate-spin" /> : "Verify OTP"}
        </Button>
      </form>
      <div className="mt-4 text-center text-sm">
        Already have an account? <Link href="/">Sign in</Link>
      </div>
    </div>
  );
};

export default VerifyOtpCard;
