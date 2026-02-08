"use client";
import { useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";
import { requestResetPassword, resetPassword } from "@/utils/api"; // Mock API calls
import { useToast } from "@/hooks/use-toast";
import Link from "next/link";
import { useRouter } from "next/navigation";

const ResetPasswordCard = () => {
  const [step, setStep] = useState(1); // Step state
  const [email, setEmail] = useState(""); // Email state
  const [otp, setOtp] = useState(""); // OTP state
  const [newPassword, setNewPassword] = useState(""); // New password state
  const [loading, setLoading] = useState(false); // Loading state
  const { toast } = useToast(); // Toast for notifications
const router = useRouter();
  // Handle request to reset password (Step 1)
  const handleRequestReset = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await requestResetPassword(email); // API call to request reset
      if (response.status === "success") {
        toast({ title: "Success", description: "OTP sent to your email." });
        setStep(2); // Move to step 2
      } else {
        toast({ title: "Error", description: "Failed to send OTP." });
      }
    } catch (error) {
      toast({ title: "Error", description: "Something went wrong!" });
    }
    setLoading(false);
  };

  // Handle password reset (Step 2)
  const handleResetPassword = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await resetPassword( email, otp, newPassword ); // API call to reset password
      if (response.status === "success") {
        toast({ title: "Success", description: "Password reset successfully!" });
        router.push("/")
      } else {
        toast({ title: "Error", description: "Failed to reset password." });
      }
    } catch (error) {
      toast({ title: "Error", description: "Something went wrong!" });
    }
    setLoading(false);
  };

  return (
    <div className="mx-auto grid w-[350px] gap-6">
      <div className="grid gap-2 text-center">
        <h1 className="text-3xl font-bold">
          {step === 1 ? "Request Password Reset" : "Reset Password"}
        </h1>
        <p className="text-muted-foreground">
          {step === 1
            ? "Enter your email to receive a reset link."
            : "Enter the OTP and your new password."}
        </p>
      </div>

      {step === 1 && (
        <form onSubmit={handleRequestReset} className="grid gap-4">
          <div className="form-group">
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              type="email"
              placeholder="Enter your email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>
          <Button type="submit" className="w-full">
            {loading ? <Loader2 className="animate-spin" /> : "Request OTP"}
          </Button>
        </form>
      )}

      {step === 2 && (
        <form onSubmit={handleResetPassword} className="grid gap-4">
          <div className="form-group">
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              type="email"
              value={email}
              readOnly
              className="bg-gray-100 cursor-not-allowed"
            />
          </div>
          <div className="form-group">
            <Label htmlFor="otp">OTP</Label>
            <Input
              id="otp"
              type="text"
              placeholder="Enter the OTP"
              value={otp}
              onChange={(e) => setOtp(e.target.value)}
              required
            />
          </div>
          <div className="form-group">
            <Label htmlFor="newPassword">New Password</Label>
            <Input
              id="newPassword"
              type="password"
              placeholder="Enter your new password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              required
            />
          </div>
          <Button type="submit" className="w-full">
            {loading ? <Loader2 className="animate-spin" /> : "Reset Password"}
          </Button>
        </form>
      )}

      <div className="mt-4 text-center text-sm">
        Remember your password?{" "}
        <Link href="/sign-in" className="underline">
          Sign in
        </Link>
      </div>
    </div>
  );
};

export default ResetPasswordCard;
