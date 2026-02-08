"use client";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Loader2 } from "lucide-react";
import Link from "next/link";
import { useToast } from "@/hooks/use-toast";
import { generateOTP, loginUser } from "@/utils/api";
import useStore from "@/store/store";
import { useRouter } from "next/navigation";

const LoginCard = () => {
  const [emailId, setEmailId] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();
  const { setToken, setEmail } = useStore();
  const router = useRouter();

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const out = await loginUser(emailId, password);
        if (out.data._doc.role === "user") {
          setToken(out?.data?.token);
          router.push("/");
          setLoading(false);
        } else if (out.data._doc.role === "admin") {
          setToken(out?.data?.token);
          router.push("/admin");
          setLoading(false);
        }
      setLoading(false);
    } catch (error) {
      toast({
        title: "Failed",
        description: "Try again.",
      });
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="mx-auto grid w-[350px] gap-6">
        <div className="grid gap-2 text-center">
          <h1 className="text-3xl font-bold">Login</h1>
          <p className="text-balance text-muted-foreground">
            Enter your email below to login to your account
          </p>
        </div>
        <form onSubmit={handleLogin} className="grid gap-4">
          <div className="grid gap-2">
            <Label htmlFor="emailId">Email</Label>
            <Input
              className="h-12"
              id="emailId"
              type="email"
              placeholder="m@example.com"
              required
              value={emailId}
              onChange={(e) => setEmailId(e.target.value)}
            />
          </div>
          <div className="grid gap-2">
            <div className="flex items-center">
              <Label htmlFor="password">Password</Label>
              <Link
                href="/reset-password"
                className="ml-auto inline-block text-sm underline"
              >
                Forgot your password?
              </Link>
            </div>
            <Input
              id="password"
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          <Button type="submit" className="w-full">
            {loading ? <Loader2 className="animate-spin" /> : "Sign in"}
          </Button>
        </form>
        <div className="mt-4 text-center text-sm">
          Don&apos;t have an account?{" "}
          <Link href="/sign-up" className="underline">
            Sign up
          </Link>
        </div>
      </div>
    </div>
  );
};

export default LoginCard;
